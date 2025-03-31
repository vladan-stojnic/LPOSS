# LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation

This repository contains the code for the paper Vladan Stojnić, Yannis Kalantidis, Jiří Matas, Giorgos Tolias, ["LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation"](http://arxiv.org/abs/2503.19777), In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

<div align="center">
    
[![arXiv](https://img.shields.io/badge/arXiv-2503.19777-b31b1b.svg)](http://arxiv.org/abs/2503.19777) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/stojnvla/LPOSS)

</div>

## Demo

The demo of our method is available at [<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" height=20px> huggingface spaces](https://huggingface.co/spaces/stojnvla/LPOSS).

## Setup

Setup the conda environment:
```
# Create conda environment
conda create -n lposs python=3.9
conda activate lposs
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
Install MMCV and MMSegmentation:
```
pip install -U openmim
mim install mmengine    
mim install "mmcv-full==1.6.0"
mim install "mmsegmentation==0.27.0"
```
Install additional requirements:
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install kornia cupy-cuda11x ftfy omegaconf open_clip_torch==2.26.1 hydra-core wandb
```

## Datasets

We use 8 benchmark datasets: PASCAL VOC20, PASCAL Context59, COCO-Object, PASCAL VOC, PASCAL Context, COCO-Stuff, Cityscapes, and ADE20k.

To run the evaluation, download and set up PASCAL VOC, PASCAL Context, COCO-Stuff164k, Cityscapes, and ADE20k datasets following ["MMSegmentation"](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html) data preparation document.

COCO-Object dataset uses only object classes from COCO-Stuff164k dataset by collecting instance segmentation annotations. Run the following command to convert instance segmentation annotations to semantic segmentation annotations:

```
python tools/convert_coco.py data/coco_stuff164k/ -o data/coco_stuff164k/
```

## Running

The provided code can be run using follwing commands:

LPOSS:
```
torchrun main_eval.py lposs.yaml --dataset {voc, coco_object, context, context59, coco_stuff, voc20, ade20k, cityscapes} [--measure_boundary]
```

LPOSS+:
```
torchrun main_eval.py lposs_plus.yaml --dataset {voc, coco_object, context, context59, coco_stuff, voc20, ade20k, cityscapes} [--measure_boundary]
```

## Citation

```
@InProceedings{stojnic2025_lposs,
    author    = {Stojni\'c, Vladan and Kalantidis, Yannis and Matas, Ji\v{r}\'i  and Tolias, Giorgos},
    title     = {LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

## Acknowledgments

This repository is based on ["CLIP-DINOiser: Teaching CLIP a few DINO tricks for Open-Vocabulary Semantic Segmentation"](https://github.com/wysoczanska/clip_dinoiser). Thanks to the authors!
