#Dice损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


class TanimotoLoss(nn.Module):
    def __init__(self):
        super(TanimotoLoss, self).__init__()

    def tanimoto(self, x, y):
        # smooth = 1e-5
        # wi = y.sum(1)
        # yy = (1 - y )
        # vi = yy.sum(1)
        # wi = vi / (wi + vi)
        # vi = wi / (wi + vi)
        # wv = y.transpose(1,0) * wi  + yy.transpose(1,0) * vi
        # wv = wv.transpose(1,0)
        # return ((x * y * wv).sum(1) + smooth) /(((x * x + (y * y) - (x * y))* wv).sum(1) + smooth)
        return (x * y).sum(1) / (x * x + (y * y) - (x * y)).sum(1)

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


class BatchSoftDiceLoss(nn.Module):

    def __init__(self,
                 p=1,
                 smooth=1,
                 weight=None,
                 ignore_lb=255):
        super(BatchSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        logitsvv = 1 - logits
        logits = torch.cat((logitsvv, logits), 1)
        label = label.squeeze(1)
        # overcome ignored label
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs*lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p)+lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer)
        denom = torch.sum(denom)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)
        return loss

class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

if __name__ == "__main__":
    # loss = TanimotoLoss()
    # predict = torch.randn(2, 1, 256, 256)
    # target = torch.randn(2, 1, 256, 256)
    #
    # score = loss(predict, target)
    # print(score)

    criteria = FocalLoss()
    logits = torch.randn(8, 1, 384, 384)
    lbs = target = torch.randint(0, 1, (8, 1, 384, 384))
    loss = criteria(logits, lbs)
    print(loss)

