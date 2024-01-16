import torch.nn as nn, torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import BCELoss


class LovaszSoftmaxV1(nn.Module):
    '''
    This is used in the boundry-category classification case
    '''
    def __init__(self, reduction='mean', ignore_index=-100):
        super(LovaszSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.lb_ignore = ignore_index

    def forward(self, logits, label):

        # overcome ignored label
        logitsvv = 1 - logits
        logits = torch.cat((logitsvv, logits), 1)
        n, c, h, w = logits.size()
        logits = logits.transpose(0, 1).reshape(c, -1).float() # use fp32 to avoid nan
        label = label.view(-1)

        idx = label.ne(self.lb_ignore).nonzero(as_tuple=False).squeeze()
        probs = logits.softmax(dim=0)[:, idx]

        label = label[idx]
        lb_one_hot = torch.zeros_like(probs).scatter_(
                0, label.unsqueeze(0), 1).detach()

        errs = (lb_one_hot - probs).abs()
        errs_sort, errs_order = torch.sort(errs, dim=1, descending=True)
        n_samples = errs.size(1)

        # lovasz extension grad
        with torch.no_grad():
            #  lb_one_hot_sort = lb_one_hot[
            #      torch.arange(c).unsqueeze(1).repeat(1, n_samples), errs_order
            #      ].detach()
            lb_one_hot_sort = torch.cat([
                lb_one_hot[i, ord].unsqueeze(0)
                for i, ord in enumerate(errs_order)], dim=0)
            n_pos = lb_one_hot_sort.sum(dim=1, keepdim=True)
            inter = n_pos - lb_one_hot_sort.cumsum(dim=1)
            union = n_pos + (1. - lb_one_hot_sort).cumsum(dim=1)
            jacc = 1. - inter / union
            if n_samples > 1:
                jacc[:, 1:] = jacc[:, 1:] - jacc[:, :-1]

        losses = torch.einsum('ab,ab->a', errs_sort, jacc)

        if self.reduction == 'sum':
            losses = losses.sum()
        elif self.reduction == 'mean':
            losses = losses.mean()
        return losses


class TanimotoLoss(nn.Module):
    def __init__(self):
        super(TanimotoLoss, self).__init__()

    def tanimoto(self, x, y):
        return (x * y).sum(1) / ((x * x).sum(1) + (y * y).sum(1) - (x * y).sum(1))

    def forward(self, pre, tar):
        '''
        pre and tar must have same shape. (N, C, H, W)
        '''
        # 获取每个批次的大小 N
        N = tar.size()[0]
        # 将宽高 reshape 到同一纬度
        input_flat = pre.view(N, -1)
        targets_flat = tar.view(N, -1)

        t = 1 - self.tanimoto(input_flat, targets_flat)
        loss = t
        # t_ = self.tanimoto(1 - input_flat, 1 - targets_flat)
        # loss = t + t_
        loss = loss.mean()
        return loss


def metric_tani(predb, yb):
    return TanimotoLoss()(predb, yb)


def metric_mse(predb, yb):
    return F.mse_loss(predb, yb)


def metric_ac(predb, yb):
    predb = (predb > 0.5).float()
    acc = (predb == yb).float().mean()
    return acc


def metric_f1(predb, yb):
    epsilon = 1e-7

    predb = (predb > 0.5).float()

    tp = (yb * predb).sum().to(torch.float32)
    tn = ((1 - yb) * (1 - predb)).sum().to(torch.float32)
    fp = ((1 - yb) * predb).sum().to(torch.float32)
    fn = (yb * (1 - predb)).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def metric_bce(predb, yb):
    return BCELoss()(predb, yb)


def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
