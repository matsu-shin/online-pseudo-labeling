
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, metric, eps=1e-8):
        super().__init__()
        self.metric = metric
        self.eps = eps

    def forward(self, input, target):
        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.sum(loss, dim=-1).mean()
        return loss
