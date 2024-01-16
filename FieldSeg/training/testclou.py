%%writefile {PACKAGE_PATH}/train.py

import os
import shutil
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import config
from datasets import *
from model import UNet
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", type=str, default='Inputs dir')
parser.add_argument("--checkpointDir", type=str, default='Checkpoint dir')

args = parser.parse_args()
print("** InputsDir:", args.inputs)
print("** CheckpointDir:", args.checkpointDir)

tmp_cred_dict = load_tmp_cred_info(".tmp_info")

import torch

print("** torch version:", torch.__version__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("** The model will be running on", DEVICE, "device")

# create local tmp dir
os.makedirs(config.PAI_LOCAL_TMP_DIR, exist_ok=True)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


## metric ACC
import torch.nn.functional as F
from torch.nn import BCELoss
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



###### IMPORT Training Data
from dataset_INT_aug import DatasetGF2
from dataset_ah import DatasetAH

def dataset_metric_recoder(model, dataset, metrics, metrics_names, global_step):
    arr_metrics = np.empty((0, len(metrics)))
    with torch.no_grad():
        for step, data in enumerate(dataset, 0):
            image, mask = data
            image = image.to(torch.float32).cuda()
            mask = mask.to(torch.float32).cuda()
            outputs = model(image)

            metrics_out = [metric(outputs, mask).cpu() for metric in metrics]
            metrics_out = np.array(metrics_out)
            arr_metrics = np.vstack((arr_metrics, metrics_out))

    arr_metrics = np.mean(arr_metrics, axis=0)

    print("Epoch {}: ".format(global_step), end=' ')
    for i in range(len(metrics)):
        print(metrics_names[i], ":", arr_metrics[i], end='; ')
    print()

    return arr_metrics


# Training model
def train():
    datasets = (DatasetAH())
    dataset = DatasetGF2(
        datasets
    )
    dataset_tra, dataset_val = dataset.dataset_split(0.9)
    dataset_val.training = False  # Validation Set上不执行数据增强


    epoch_val = 1  # 验证用的epoch周期
    es_epoch = 20  # 模型早停指标，若连续es_epoch个周期模型未超越最优精度，则停止优化

    model = UNet().cuda()
    loss_fn = TanimotoLoss()  # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

    # Loading 以10%的比例，随机划分训练集和验证集
    trainloader = DataLoader(dataset_tra, batch_size=config.BATCH_SIZE, shuffle = True, num_workers = 0)
    valloader = DataLoader(dataset_val, batch_size=config.BATCH_SIZE, shuffle = True, num_workers = 0)

    # 定义验证用的metrics函数和其再tensorboard中显示的名字
    metrics_fs = [metric_f1, metric_ac, metric_mse,
                  metric_bce, metric_tani]
    metrics_names = ['F1', 'AC', 'MSE', 'BCE', 'Tani']

    # 初始化最佳指标
    metrics_best = 0  # 会将val集验证的metrics的第一个metric当成model saving指标
    es_epoch_i = 0  # 初始化早停指标。指标为，

    for epoch in range(config.NUM_EPOCHES):
        # Training
        model.train()
        for step, data in enumerate(trainloader, 0):
            image, mask = data
            image = image

            image = image.to(torch.float32).cuda()
            mask = mask.to(torch.float32).cuda()

            optimizer.zero_grad()
            outputs = model(image)

            loss = loss_fn(outputs, mask)
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print('Current epoch-step: {}-{}  Loss: {}  AllocMem (Mb): {}'
                      .format(epoch, step, loss, torch.cuda.memory_allocated() / 1024 / 1024))
        scheduler.step()

        # Vaildation
        model.eval()
        if epoch % epoch_val == 0:

            # 分别对训练集和测试集进行验证
            dataset_metric_recoder(model, trainloader, metrics_fs, metrics_names, epoch)

            val_metrics = dataset_metric_recoder(model, valloader, metrics_fs, metrics_names, epoch)

            if val_metrics[0] > metrics_best:
                save_model(model, os.path.join(config.PAI_LOCAL_TMP_DIR, config.OUTPUT_MODEL_FILE_NAME))
                metrics_best = val_metrics[0]
                es_epoch_i = 0

            # Early Stop
            es_epoch_i += 1
            if es_epoch_i == es_epoch:
                print("Early Stop Happen. Current Epoch is: ", epoch)
                break

    # 将tmp目录中的 bestModel pth文件 传到OSS的checkpoint路径中  TMP_ACCESS_ID, TMP_ACCESS_SEC, config.OSS_HOST, TMP_BUCKET_NAME
    upload_object_to_oss(tmp_cred_dict.get('ossStsAccessKeyId'),
                         tmp_cred_dict.get('ossStsAccessKeySecret'),
                         tmp_cred_dict.get('ossStsAccessSecurityToken'),
                         config.OSS_HOST,
                         tmp_cred_dict.get('ossBucketName'),
                         os.path.join(tmp_cred_dict.get('userWorkDir'), config.OSS_CHECKPOINT_DIR,
                                      config.OUTPUT_MODEL_FILE_NAME),
                         os.path.join(config.PAI_LOCAL_TMP_DIR, config.OUTPUT_MODEL_FILE_NAME))

    # 清理临时文件夹
    shutil.rmtree(config.PAI_LOCAL_TMP_DIR)

    print("** Training process done .\n")


if __name__ == "__main__":
    print("** Start training process.\n")

    train()