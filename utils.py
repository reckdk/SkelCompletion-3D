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
