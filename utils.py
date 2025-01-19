import os
import pandas as pd
import numpy as np
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss
from torch import sum as torchsum
from torch import where as torchwhere

import gc


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config = json.load(config_file)
    #config = Bunch(config)
    return config


def window_search_3D(img, masks, BMAmask, imgshape, patch_size, stride):
    '''
    Sliding window for 3D patches extraction.
    img in shape (Pz,Py,Px). 
    masks in shape (3,Pz,Py,Px) (FG, MG, BG).
    BMAmask in shape (Pz,Px).

    Each volume: 30G RAM, 3.5s --> 6G, 4s
    '''
    img_z, img_y, img_x = imgshape
    patch_z, patch_y, patch_x = patch_size
    stride_z, stride_x = stride if len(stride)==2 else [stride, stride]

    # The number of patches cropped from the image.
    #patch_num_z = int(np.ceil((img_z-patch_z) / stride_z)) # Incorrect.
    #patch_num_x = int(np.ceil((img_x-patch_x) / stride_x))
    patch_num_z = int(np.ceil((img_z-patch_z) / stride_z)) + 1
    patch_num_x = int(np.ceil((img_x-patch_x) / stride_x)) + 1

    # Placeholder.
    img_patch_array = np.zeros((patch_num_z*patch_num_x, patch_z, patch_y, patch_x), dtype=np.float32)
    masks_patch_array = np.zeros((patch_num_z*patch_num_x, 3, patch_z, patch_y, patch_x), dtype=np.bool8)
    BMAmask_patch_array = np.zeros((patch_num_z*patch_num_x, patch_z, 1, patch_x), dtype=np.bool8)

    for idx_z in range(patch_num_z):
        # If cropping outside the boundary, shifting the window back to the boundary.
        start_z = (stride_z*idx_z) if (stride_z*idx_z + patch_z) <= img_z else (img_z-patch_z)
        for idx_x in range(patch_num_x):
            start_x = (stride_x*idx_x) if (stride_x*idx_x + patch_x) <= img_x else (img_x-patch_x)
            img_patch_array[idx_z*patch_num_x + idx_x] = img[
                start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)]
            masks_patch_array[idx_z*patch_num_x + idx_x] = masks[
                :, start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)]
            BMAmask_patch_array[idx_z*patch_num_x + idx_x] = BMAmask[
                start_z : (start_z+patch_z), np.newaxis, start_x : (start_x+patch_x)]

    return img_patch_array, masks_patch_array, BMAmask_patch_array


def patch_reconstruction_3D(img_patch_array, imgshape, patch_size, stride):
    '''
    The inverse of sliding window for 3D patches.
    Reconstruct the origial 3D volume from patches.

    img_patch_array in shape (N,Pz,Py,Px).
    imgshape in shape (img_z, img_y, img_x)
    Return img_stack in original image shape.

    Time cost: 9.45s.
    '''
    img_z, img_y, img_x = imgshape
    img_n = img_patch_array.shape[0]
    patch_z, patch_y, patch_x = patch_size
    stride_z, stride_x = stride if len(stride)==2 else [stride, stride]

    # The number of patches cropped from the image.
    patch_num_z = int(np.ceil((img_z-patch_z) / stride_z)) + 1
    patch_num_x = int(np.ceil((img_x-patch_x) / stride_x)) + 1

    assert img_n == patch_num_z*patch_num_x, 'Input patch number inconsistent.'

    # Placeholder.
    img_stack = np.zeros((img_z, img_y, img_x), dtype=np.float32)
    img_stack_count = np.zeros((img_z, img_y, img_x), dtype=np.float32)

    for idx_z in range(patch_num_z):
        # If cropping outside the boundary, shifting the window back to the boundary.
        start_z = (stride_z*idx_z) if (stride_z*idx_z + patch_z) <= img_z else (img_z-patch_z)
        for idx_x in range(patch_num_x):
            start_x = (stride_x*idx_x) if (stride_x*idx_x + patch_x) <= img_x else (img_x-patch_x)
            img_stack[start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)] += img_patch_array[idx_z*patch_num_x + idx_x]
            img_stack_count[start_z : (start_z+patch_z), :, start_x : (start_x+patch_x)] += 1

    assert img_stack_count.min() != 0, 'Every voxel should be predicted at least once.'
    # Normalization.
    img_stack = img_stack / img_stack_count

    return img_stack


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        #input = torch.sigmoid(input)
        input = input.flatten()
        target = target.flatten()
        
        intersection = (input * target).sum()
        dice = (2.*intersection + self.epsilon)/(input.sum() + target.sum() + self.epsilon)

        return 1 - dice

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, epsilon=1, reduction='mean'):
    #def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, inputs, targets):

        #inputs = torch.sigmoid(inputs)
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.epsilon) / (TP + self.alpha*FP + self.beta*FN + self.epsilon)  
        
        return 1 - Tversky

class L1LossForDiceBG(L1Loss):
    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, inputs, targets, roi_mask):
        l1 = super().forward(inputs, targets).view(-1)
        fg_mask = (roi_mask[:, 1]==1).flatten()
        return l1[~fg_mask].mean()

class NaiveL1Loss(L1Loss):
    def __init__(self):
        super().__init__(epsilon=1, reduction="none")
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        l1 = super().forward(inputs, targets).view(-1)
        return l1.mean()

def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.9,
    gamma: float = 1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    #ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss



class MaskedDiceLoss(nn.Module):
    def __init__(self):
        super(MaskedDiceLoss, self).__init__()

    def forward(self, input, target, roi_mask, epsilon=1):
        input = torch.sigmoid(input) # For convenient transfering from EDT branch.
        input = input.flatten()
        target = target.flatten()
        fg_mask = roi_mask[:, 1:2].flatten()
        
        intersection = input * target
        dice = (2.*intersection.sum() + epsilon) / (input.sum() + target.sum() + epsilon)
        dice_roi = (2.*torch.sum(intersection*fg_mask) + epsilon) / (
            torch.sum(input*fg_mask) + torch.sum(target*fg_mask) + epsilon)

        return 1 - dice, 1 - dice_roi

#PyTorch
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class MaskedWeightedL1Loss(L1Loss):
    def __init__(self, bg_weight=1.0):
        super().__init__(reduction="none")
        self.bg_weight = bg_weight

    def forward(self, input, target, seg_mask):
        l1 = super().forward(input, target)

        return (l1[seg_mask==1].mean()
                + l1[seg_mask!=1].mean() * self.bg_weight)


class WeightedL1Loss(L1Loss):
    '''
    Return FG/BG-weighted maskedL1Loss regardless of batchsize.
    As validation metric, not training loss.
    '''
    def __init__(self, bg_weight=1.0): #reduction='mean')?
        super().__init__(reduction="none")
        self.bg_weight = bg_weight

    def forward(self, inputs, targets, roi_mask):
        l1 = super().forward(inputs, targets).view(-1)
        # At this time, we count on both skel and BG.
        fg_mask = (roi_mask[:, 2]==1).view(-1)
        return (l1[fg_mask].mean() + l1[~fg_mask].mean()*self.bg_weight).mean()

class WeightedMSELoss(MSELoss):
    def __init__(self, bg_weight=1.0):
        super().__init__(reduction="none")
        self.bg_weight = bg_weight

    def forward(self, inputs, targets, roi_mask):
        loss_mse = super().forward(inputs, targets).view(-1)
        # At this time, we count on both skel and BG.
        fg_mask = (roi_mask[:, 2]==1).view(-1)
        return (loss_mse[fg_mask].mean() + loss_mse[~fg_mask].mean()*self.bg_weight).mean()


class ConnectionMetric(nn.Module):
    '''
    Validation metric, not training loss.
    ---
    Input

    `input`: (B,1,P,P,P)
    `target`: (B,1,P,P,P)
    `roi_mask`: (B,3,P,P,P)

    ---
    Return
    
    `Recall`: skel_TP/|skel| # Important.
    #`Precision`: skel_TP/(skel_TP+skel_FP) # Should be low.
    #`Accuracy`: (skel_TP+skel_TF)/RoISize? or PatchSize
    `success_rate`: if two EPs are reconnected with prediction.

    Note: due to returned values are too small, use following zoom-in method:
    1e4*(1-ret)

    '''
    def __init__(self, bg_weight=1, epsilon=1):
        super(ConnectionMetric, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target, roi_mask):
        input = input.flatten(1) # (B,-1)
        target = target.flatten(1)
        roi_mask = roi_mask.flatten(2) # (B,3,-1)

        pos = input > 0.5
        pos_gt = target * roi_mask[:, 2]
        recall = (pos_gt*pos + self.epsilon) / (pos_gt + self.epsilon)
        success_rate = (recall==1).float().mean()

        return recall.mean(), success_rate
        #return 1e4 * (1-recall.mean()), 1e4 * (1-success_rate)

class MaskedL1Loss(L1Loss):
    '''
    Return maskedL1Loss regardless of batchsize.
    This loss is customized for mask or EDT tasks.

    At this time, we count on
    (1) specific RoI region.
    (2) (1) and BG.
    (2) is better than (1) on EDT.

    DO NOT use for other tasks.
    '''
    def __init__(self, epsilon=1e-4):
        super().__init__(reduction="none")
        self.epsilon = epsilon

    def ed_forwarddd(self, input, target, roi_mask):
        l1 = super().forward(input, target).view(-1)
        fg_mask = (roi_mask[:, 1]==1).view(-1)
        #return l1[fg_mask].mean()
        return l1[fg_mask].mean(), l1[~fg_mask].mean()

    def forward(self, input, target, roi_mask):
        l1 = super().forward(input, target)

        l1_fg = ((l1 * roi_mask[:, 1:2]).sum([1,2,3,4]) + self.epsilon) / (
            roi_mask[:, 1:2].sum([1,2,3,4]) + self.epsilon)

        roi_nonfg = 1 - roi_mask[:, 1:2]
        l1_nonfg = ((l1 * roi_nonfg).sum([1,2,3,4]) + self.epsilon) / (
            roi_nonfg.sum([1,2,3,4]) + self.epsilon)

        #return l1_fg.mean()
        return l1_fg.mean(), l1_nonfg.mean()


class ConnLoss(L1Loss):
    '''
    L_conn
    Return L1Loss penalizing more on missed part.
    '''
    def __init__(self, bg_weight=1.0, fg_th=0.5, penalty_FP=2): # [Need FT]
        super().__init__(reduction="none")
        self.bg_weight = bg_weight
        self.fg_th = fg_th
        self.penalty_FP = penalty_FP

    def forward(self, input, target, roi_mask):
        l1 = super().forward(input, target).view(-1)
        # At this time, we only count on skel RoI.
        fg_mask = (roi_mask[:, 2]==1).view(-1)
        loss_naive = (l1[fg_mask].mean() + l1[~fg_mask].mean()*self.bg_weight).mean()

        l1_fg = l1[fg_mask]
        cond_FN = l1_fg > self.fg_th
        loss_connection = torch.where(cond_FN, self.penalty_FP + l1_fg, l1_fg)

        return loss_connection.mean(), loss_naive
