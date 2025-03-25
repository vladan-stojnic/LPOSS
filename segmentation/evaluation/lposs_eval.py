from itertools import product
import logging
import math
import os
import time
from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F

log = logging.getLogger(__name__)
from mmseg.ops import resize
from mmseg.models import EncoderDecoder

from cupyx.scipy.sparse import csr_matrix, diags, eye, coo_matrix
from cupyx.scipy.sparse import linalg as s_linalg
import cupy as cp

import faiss
import numpy as np
import faiss.contrib.torch_utils
from PIL import Image
import numpy as np
from kornia.color import rgb_to_lab


def reshape_windows(x):
    height_width = [(y.shape[0], y.shape[1]) for y in x]
    dim = x[0].shape[-1]
    x = [torch.reshape(y, (-1, dim)) for y in x]
    # x = x.reshape((-1, ))
    
    return torch.cat(x, dim=0), height_width


def normalize_connection_graph(G):
    W = csr_matrix(G)
    W = W - diags(W.diagonal(), 0)
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = cp.array(1.0 / cp.sqrt(S))
    D[cp.isnan(D)] = 0
    D[cp.isinf(D)] = 0
    D_mh = diags(D.reshape(-1), 0)
    Wn = D_mh * W * D_mh
    return Wn


def get_lposs_laplacian(feats, locations, height_width, sigma=0.0, pix_dist_pow=2, k=100, gamma=1.0, alpha=0.95, patch_size=16):
    idx_window = torch.cat([window * torch.ones((h*w, ), device=feats.device, dtype=torch.int64) for window, (h, w) in enumerate(height_width)])
    idx_h = torch.cat([torch.arange(h).view(-1,1).repeat(1, w).flatten() for h, w in height_width]).to(feats.device)
    idx_w = torch.cat([torch.arange(w).view(1,-1).repeat(h, 1).flatten() for h, w in height_width]).to(feats.device)
    loc_h = locations[idx_window, 0] + (patch_size // 2) + idx_h * patch_size
    loc_w = locations[idx_window, 2] + (patch_size // 2) + idx_w * patch_size
    locs = torch.stack((loc_h, loc_w), 1)
    locs = torch.unsqueeze(locs, 0)
    dist = torch.cdist(locs, locs, p=2)
    dist = dist[0, ...]
    dist = dist ** pix_dist_pow
    geometry_affinity = torch.exp(-sigma * dist)

    N = feats.shape[0]
    
    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()
    res.setTempMemory(0)
    sims, ks = faiss.knn_gpu(res, feats, feats, k, metric=faiss.METRIC_INNER_PRODUCT)

    sims[sims < 0] = 0
    sims = sims ** gamma
    geometry_affinity = geometry_affinity.gather(1, ks).flatten()
    sims = sims.flatten()
    sims = sims * geometry_affinity
    ks = ks.flatten()
    rows = torch.arange(N).repeat_interleave(k)
    
    W = csr_matrix(
        (cp.asarray(sims), (cp.asarray(rows), cp.asarray(ks))),
        shape=(N, N),
    )
    W = W + W.T
    Wn = normalize_connection_graph(W)
    L = eye(Wn.shape[0]) - alpha * Wn

    return L


def dfs_search(L, Y, tol=1e-6, maxiter=10):
    out = s_linalg.cg(L, Y, tol=tol, maxiter=maxiter)[0]

    return out


def perform_lp(L, preds):
    lp_preds = cp.zeros(preds.shape)
    preds = cp.asarray(preds)
    for cls_idx, y_cls in enumerate(preds.T):
        Y = y_cls
        lp_preds[:, cls_idx] = dfs_search(L, Y)
    lp_preds = torch.as_tensor(lp_preds, device="cuda")

    return lp_preds


def get_pixel_connections(img, neigh=1):
    img = img[0, ...]
    img_lab = rgb_to_lab(img)
    img_lab = img_lab.permute((1, 2, 0))
    img_lab /= torch.tensor([100, 128, 128], device=img.device) # project Lab values to 0-1 range
    img_h, img_w, _ = img_lab.shape
    img_lab = img_lab.reshape((img_h*img_w, -1))

    idx = torch.arange(img_h * img_w).to(img.device)
    loc_h = idx // img_w
    loc_w = idx % img_w
    locs = torch.stack((loc_h, loc_w), 1)
    
    rows, cols = [], []

    for mov in product(range(-neigh, neigh+1), range(-neigh, neigh+1)):
        if mov[0] == 0 and mov[1] == 0:
            continue
        new_locs = locs + torch.tensor(mov).to(img.device)
        mask = torch.logical_and(torch.logical_and(torch.logical_and(new_locs[:, 0] >= 0, new_locs[:, 1] >= 0), new_locs[:, 0] < img_h), new_locs[:, 1] < img_w)
        rows.append(torch.where(mask)[0])
        col = new_locs[mask, :]
        col = col[:, 0] * img_w + col[:, 1]
        cols.append(col)

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    pixel_pixel_data = ((img_lab[rows, :] - img_lab[cols, :]) ** 2).sum(dim=-1)

    return rows, cols, pixel_pixel_data, locs


def get_lposs_plus_laplacian(img, preds, tau=0.1, neigh=6, alpha=0.95):
    rows, cols, pixel_pixel_data, locs = get_pixel_connections(img, neigh=neigh)
    pixel_pixel_data = torch.sqrt(pixel_pixel_data)
    pixel_pixel_data = torch.exp(-pixel_pixel_data / tau)
    
    N = preds.shape[0]
    rows = cp.asarray(rows)
    cols = cp.asarray(cols)
    data = cp.asarray(pixel_pixel_data)
    W = csr_matrix(
        (data, (rows, cols)),
        shape=(N, N),
    )

    Wn = normalize_connection_graph(W)
    L = eye(Wn.shape[0]) - alpha * Wn

    return L


class LPOSS_Infrencer(EncoderDecoder):
    def __init__(
            self,
            model,
            config,
            num_classes,
            test_cfg=dict(),
            **kwargs,
    ):
        super(EncoderDecoder, self).__init__()
        self.mode = test_cfg['mode']
        self.num_classes = num_classes
        self.model = model
        self.test_cfg = test_cfg
        self.align_corners = False
        self.config = config

    @torch.no_grad()
    def encode_decode(self, img, meta_data):
        dino_feats, clip_feats, clf = self.model(img)
        return dino_feats, clip_feats, clf
    

    @torch.no_grad()
    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict], rescale) -> Tensor:
        _, _, h_img, w_img = inputs.size()
        img_dino_feats, img_clip_feats, img_clf = self.encode_decode(inputs, batch_img_metas)
        if img_clip_feats.shape[1] != img_dino_feats.shape[1] or img_clip_feats.shape[2] != img_dino_feats.shape[2]:
            img_clip_feats = F.interpolate(img_clip_feats.permute(0, 3, 1, 2), size=(img_dino_feats.shape[1], img_dino_feats.shape[2]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        img_dino_feats = img_dino_feats[0, ...]
        img_clip_feats = img_clip_feats[0, ...]
        clf = img_clf

        num_classes = clf.shape[0]
        
        h, w, _ = img_dino_feats.shape
        img_dino_feats = img_dino_feats.reshape((h*w, -1))
        img_clip_feats = img_clip_feats.reshape((h*w, -1))
        
        dino_feats = F.normalize(img_dino_feats, p=2, dim=-1)
        clip_feats = F.normalize(img_clip_feats, p=2, dim=-1)
        clip_preds = clip_feats @ clf.T

        L = get_lposs_laplacian(dino_feats, torch.zeros((1, 4)).to(dino_feats.device), [(h, w)], sigma=self.config.sigma, pix_dist_pow=self.config.pix_dist_pow, k=self.config.k, gamma=self.config.gamma, alpha=self.config.alpha, patch_size=self.config.model.vit_patch_size)
        
        lp_preds = perform_lp(L, clip_preds)

        preds = lp_preds.reshape((h, w, num_classes))
        preds = preds.unsqueeze(0)
        preds = preds.permute((0, 3, 1, 2))

        if self.config.pixel_refine:
            preds = resize(preds, size=(h_img, w_img), mode='bilinear', align_corners=self.align_corners)
            preds = preds[0, ...]
            preds = preds.permute((1, 2, 0))
            preds = preds.reshape((h_img*w_img, -1))
            L = get_lposs_plus_laplacian(inputs, preds, tau=self.config.tau, neigh=self.config.r // 2, alpha=self.config.alpha)
            preds = perform_lp(L, preds)
            preds = preds.reshape((h_img, w_img, num_classes)).permute((2, 0, 1)).unsqueeze(0)

        if preds.shape[1] > self.num_classes:
            preds = self.reduce_to_true_classes(preds)

        masks = resize(
                preds,
                size=batch_img_metas[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        return masks


    @torch.no_grad()
    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict], rescale) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        clf = None
        locations = inputs.new_zeros((h_grids*w_grids, 4))
        # dino_feats = []
        # clip_feats = []
        images = []

        # go over sliding windows and extract features
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                images.append(crop_img)

                # batch_img_metas[0]['img_shape'] = crop_img.shape[2:]

                # img_dino_feats, img_clip_feats, img_clf = self.encode_decode(crop_img, batch_img_metas)

                # if img_clip_feats.shape[1] != img_dino_feats.shape[1] or img_clip_feats.shape[2] != img_dino_feats.shape[2]:
                #     img_clip_feats = F.interpolate(img_clip_feats, size=(img_dino_feats.shape[1], img_dino_feats.shape[2]), mode='bilinear', align_corners=False)

                # dino_feats.append(img_dino_feats[0, ...])
                # clip_feats.append(img_clip_feats[0, ...])
                locations[h_idx*w_grids + w_idx, 0] = y1
                locations[h_idx*w_grids + w_idx, 1] = y2
                locations[h_idx*w_grids + w_idx, 2] = x1
                locations[h_idx*w_grids + w_idx, 3] = x2
                # if clf is None:
                #     clf = img_clf

        images = torch.cat(images, dim=0)
        dino_feats, clip_feats, clf = self.encode_decode(images, None)
        if clip_feats.shape[1] != dino_feats.shape[1] or clip_feats.shape[2] != dino_feats.shape[2]:
            clip_feats = F.interpolate(clip_feats.permute(0, 3, 1, 2), size=(dino_feats.shape[1], dino_feats.shape[2]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        # breakpoint()
                
        num_classes = clf.shape[0]
        
        dino_feats, height_width = reshape_windows(dino_feats)
        clip_feats, _ = reshape_windows(clip_feats)
        dino_feats = F.normalize(dino_feats, p=2, dim=-1)
        clip_feats = F.normalize(clip_feats, p=2, dim=-1)
        clip_preds = clip_feats @ clf.T
        
        L = get_lposs_laplacian(dino_feats, locations, height_width, sigma=self.config.sigma, pix_dist_pow=self.config.pix_dist_pow, k=self.config.k, gamma=self.config.gamma, alpha=self.config.alpha, patch_size=self.config.model.vit_patch_size)
        
        lp_preds = perform_lp(L, clip_preds)

        # resize and overlap window predictions
        preds = inputs.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        idx_window = torch.cat([window * torch.ones((h*w, ), device=dino_feats.device, dtype=torch.int64) for window, (h, w) in enumerate(height_width)])
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                win_id = h_idx*w_grids + w_idx
                crop_seg_logit = lp_preds[torch.where(idx_window == win_id)[0], :]
                crop_seg_logit = torch.reshape(crop_seg_logit, height_width[win_id]+(num_classes, ))
                crop_seg_logit = torch.unsqueeze(crop_seg_logit, 0)
                crop_seg_logit = torch.permute(crop_seg_logit, (0, 3, 1, 2))
                crop_seg_logit = resize(
                    input=crop_seg_logit,
                    size=(y2-y1, x2-x1),
                    mode='bilinear',
                    align_corners=False
                )
                assert crop_seg_logit.shape[2] == (y2 - y1) and crop_seg_logit.shape[3] == (x2 - x1)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat

        # apply lposs+
        if self.config.pixel_refine:
            preds = preds[0, ...]
            preds = preds.permute((1, 2, 0))
            preds = preds.reshape((h_img*w_img, -1))
            L = get_lposs_plus_laplacian(inputs, preds, tau=self.config.tau, neigh=self.config.r // 2, alpha=self.config.alpha)
            preds = perform_lp(L, preds)
            preds = preds.reshape((h_img, w_img, num_classes)).permute((2, 0, 1)).unsqueeze(0)

        # use original number of classes if class expansion was used
        if preds.shape[1] > self.num_classes:
            preds = self.reduce_to_true_classes(preds)

        masks = resize(
                preds,
                size=batch_img_metas[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return masks
    
    def reduce_to_true_classes(self, preds):
        num_background_classes = preds.shape[1] - self.num_classes + 1
        new_preds = torch.zeros((preds.shape[0], self.num_classes, preds.shape[2], preds.shape[3]), device=preds.device, dtype=preds.dtype)
        new_preds[:, 1:, :, :] = preds[:, num_background_classes:, :, :]
        new_preds[:, 0, :, :] = torch.max(preds[:, :num_background_classes, :, :], dim=1)[0]

        return new_preds