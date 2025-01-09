import torch
import torch.nn as nn
import torch.nn.functional as F
from .m5_class import M5Class as M5


class FullModelClass(nn.Module):
    def __init__(self, m5net: M5):
        super(FullModelClass, self).__init__()

        self.m5 = m5net

    # def forward(self, t_in, t_gt_kls, t_gt_rts):
    def forward(self, t_in, t_gt_kls):
        t_pr = self.m5(t_in)
        kls_count = self.m5.class_count

        # loss = FullModelClass.calc_loss(t_pr, t_gt_kls, t_gt_rts)
        loss = FullModelClass.calc_loss(t_pr, t_gt_kls, kls_count)
        accuracy = FullModelClass.calc_accuracy(t_pr, t_gt_kls, kls_count)

        return loss, accuracy

    @staticmethod
    def calc_accuracy(t_pr, t_gt, kls_count):

        # t_pr = torch.argmax(t_pr[..., 0:kls_count], axis=-1)
        # acc = (t_pr.eq(t_gt).sum() / t_gt.numel()).mean()

        return (1. - torch.abs(t_pr.view(-1) - t_gt)).mean()

    @staticmethod
    def calc_loss(t_pr, t_gt_kls, kls_count):
    # def calc_loss(t_pr, t_gt_kls, t_gt_rts):

        # loss_kls = F.cross_entropy(t_pr[..., 0:kls_count], t_gt_kls)
        # loss_rts = F.smooth_l1_loss(t_pr[..., kls_count:], t_gt_rts)

        loss_kls = ((t_pr[..., 0:kls_count] - t_gt_kls.unsqueeze(1)) ** 2)

        return loss_kls.mean()# + loss_rts
