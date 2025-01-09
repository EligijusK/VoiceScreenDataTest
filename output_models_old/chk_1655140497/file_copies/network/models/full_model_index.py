import torch
import torch.nn as nn
import torch.nn.functional as F
from .m5_index import M5Index as M5


class FullModelIndex(nn.Module):
    def __init__(self, m5net: M5):
        super(FullModelIndex, self).__init__()

        self.m5 = m5net

    def forward(self, t_in, t_gt):
        t_pr = self.m5(t_in)

        loss = FullModelIndex.calc_loss(t_pr, t_gt)

        return loss

    @staticmethod
    def calc_loss(t_pr, t_gt):
        loss = F.mse_loss(t_pr, t_gt)
        
        return loss
