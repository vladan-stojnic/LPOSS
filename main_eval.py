# ---------------------------------------------------------------------------------------------------
# LPOOS
# ---------------------------------------------------------------------------------------------------
# modified from CLIP-DINOiser (https://github.com/wysoczanska/clip_dinoiser)
# ---------------------------------------------------------------------------------------------------

import argparse
import datasets.transforms

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmseg.apis import multi_gpu_test

from helpers.logger import get_logger
from models import build_model
from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
import wandb


@torch.no_grad()
def evaluate(cfg, val_loaders, measure_boundary=False):
    logger = get_logger()
    ret = {}

    for key, loader in val_loaders.items():

        logger.info(f"### Validation dataset: {key}")
        CLASSES = loader.dataset.CLASSES
        logger.info(f"Creating model:{cfg.model.type}")
        model = build_model(cfg.model, class_names=CLASSES)
        model.cuda()
        model.device = "cuda"
        model.eval()

        miou, boundary_iou, metrics = validate_seg(cfg, cfg.evaluate.get(key), loader, model, measure_boundary=measure_boundary)
        logger.info(f"[{key}] mIoU of {len(loader.dataset)} test images: {miou:.2f}%")
        ret[f"val/{key}_miou"] = miou
        wandb.log({f"{key}_mIoU": miou})
        if measure_boundary:
            ret[f"val/{key}_boundary_iou"] = boundary_iou
            logger.info(f"[{key}] Boundary IoU of {len(loader.dataset)} test images: {boundary_iou:.2f}%")
            wandb.log({f"{key}_boundary_iou": boundary_iou})

    ret["val/avg_miou"] = np.mean([v for k, v in ret.items() if "miou" in k])
    if measure_boundary:
        ret["val/avg_boundary_iou"] = np.mean([v for k, v in ret.items() if "boundary_iou" in k])
    return ret


@torch.no_grad()
def validate_seg(config, seg_config, data_loader, model, measure_boundary=False):
    logger = get_logger()
    dist.barrier()
    model.eval()
    seg_model = build_seg_inference(
        model,
        data_loader.dataset,
        config,
        seg_config,
    )

    mmddp_model = MMDistributedDataParallel(
        seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False
    )
    mmddp_model.eval()

    results, boundary_results = multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False,
        measure_boundary=measure_boundary,
    )

    if dist.get_rank() == 0:
        if measure_boundary:
            metric = [data_loader.dataset.evaluate(results, metric="mIoU", logger=logger), data_loader.dataset.evaluate(boundary_results, metric="mIoU", logger=logger)]
        else:
            metric = [data_loader.dataset.evaluate(results, metric="mIoU", logger=logger)]
    else:
        metric = [None]

    dist.broadcast_object_list(metric)
    miou_result = metric[0]["mIoU"] * 100
    if measure_boundary:
        boundary_result = metric[1]["mIoU"] * 100
    else:
        boundary_result = None
    torch.cuda.empty_cache()
    dist.barrier()
    return miou_result, boundary_result, metric


def main(cfg, dataset, measure_boundary=False):
    mp.set_start_method("fork", force=True)
    init_dist("pytorch")
    rank, world_size = get_dist_info()
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

    dist.barrier()
    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    logger = get_logger(cfg)

    val_loaders = {}
    for key in cfg.evaluate.task:
        if key == dataset:
            loader = build_seg_dataloader(build_seg_dataset(cfg.evaluate.get(key)))
            val_loaders[key] = loader
    res = evaluate(cfg, val_loaders, measure_boundary=measure_boundary)
    logger.info(res)
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LPOSS evaluation")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--measure_boundary", action="store_true", help="measure boundary iou")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)
    wandb.init(project="LPOSS final", config=args)
    main(cfg, args.dataset, measure_boundary=args.measure_boundary)
