import cv2
import numpy as np
import torch


# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou_per_mask(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou, intersection, union, dt_boundary.sum(), gt_boundary.sum()


def boundary_iou(gt, dt, num_classes, dilation_ratio=0.02, ignore_index=255, reduce_zero_label=False):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    area_pred = np.zeros(num_classes)
    area_gt = np.zeros(num_classes)
    dt = np.copy(dt)
    gt = np.copy(gt)
    if reduce_zero_label:
        gt[gt == 0] = ignore_index
        gt -= 1
        gt[gt == ignore_index - 1] = ignore_index
    dt[gt == ignore_index] = ignore_index

    for cls_idx in (set(np.unique(gt)).union(np.unique(dt)))-set([ignore_index]):
        gt_bin = (gt == cls_idx).astype(np.uint8)
        dt_bin = (dt == cls_idx).astype(np.uint8)
        _, inter, uni, ap, agt = boundary_iou_per_mask(gt_bin, dt_bin, dilation_ratio=dilation_ratio)
        intersection[cls_idx] += inter
        union[cls_idx] += uni
        area_pred[cls_idx] += ap
        area_gt[cls_idx] += agt

    return torch.tensor(intersection), torch.tensor(union), torch.tensor(area_pred), torch.tensor(area_gt)