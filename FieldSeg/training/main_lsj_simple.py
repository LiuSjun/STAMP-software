'''
这是LSJ模型的简化版本
'''

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import torch.nn as nn
import os, torch, UNIT
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
# from model.UNet import UNet, DeepLabV3, PAN
from model.resnet_unet_0903 import UNet_Bou
from TanimotoLoss import TanimotoLoss, LovaszSoftmaxV1, FocalLoss
import model.util as util_mod
from torch.utils.tensorboard import SummaryWriter

dataset_root = r"F:\Code\CropPatch\Crop\share\data"
#更改结构，只进行分割
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter as sw

ROOT_PTN = r"D:\CropSegmentation\pths"
ROOT_RST = r"D:\CropSegmentation\result"

###### IMPORT Training Data
from dataset.dataset_INT import DatasetGF2
from dataset.datasets.dataset_ah import DatasetAH
from dataset.datasets.dataset_cd import DatasetCD
from dataset.datasets.dataset_gs import DatasetGS
from dataset.datasets.dataset_hlj import DatasetHLJ
from dataset.datasets.dataset_ms import DatasetMS
from dataset.datasets.dataset_zj import DatasetZJ
from dataset.datasets.dataset_zz import DatasetZZ

datasets = (DatasetAH(), DatasetCD(), DatasetGS(), DatasetHLJ(), DatasetMS(), DatasetZJ(), DatasetZZ())
dataset = DatasetGF2(
    datasets
)
dataset_tra, dataset_val = dataset.dataset_split(0.9)


val_percent = 0.1


def dataset_metric_recoder(model, dataset, writer: sw, writer_name, metrics, metrics_names, global_step):
    arr_metrics = np.empty((0, len(metrics)))
    with torch.no_grad():
        for step, data in enumerate(dataset, 0):
            image, mask = data
            image = image[:, :3, :, :]
            image = image.to(torch.float32).cuda()
            mask = mask.to(torch.float32).cuda()
            outputs = model(image)

            metrics_out = [metric(outputs, mask).cpu() for metric in metrics]
            metrics_out = np.array(metrics_out)
            arr_metrics = np.vstack((arr_metrics, metrics_out))

    arr_metrics = np.mean(arr_metrics, axis=0)
    writer.add_scalars(writer_name, dict(zip(metrics_names, arr_metrics)), global_step)

    print("Epoch {}: ".format(global_step), end=' ')
    for i in range(len(metrics)):
        print(metrics_names[i], ":", arr_metrics[i], end='; ')
    print()

    return arr_metrics

# def training_seg_boun():
#     spe_dir = os.path.join(dataset_root, "Spec") #Spectral Spec
#     segmentation_dir = os.path.join(dataset_root, "Semantic")   #  Boundary Semantic Label
#     dataset_extent = DatasetSegtask(spe_dir, segmentation_dir)
#     n_val = int(len(dataset_extent) * val_percent)
#     n_train = len(dataset_extent) - n_val
#     train_dataset_extent, val_dataset_extent = random_split(dataset_extent, [n_train, n_val])
#
#     model = UNet_Bou().cuda()
#     load_name = ''
#     # model_vgg.train()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if load_name:
#         model.load_state_dict(
#             torch.load(load_name, map_location=device)
#         )
#
#     loss_fn = LovaszSoftmaxV1() # TanimotoLoss() LovaszSoftmaxV1 BatchSoftDiceLoss  FocalLoss # loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
#
#     print("Data Loading...")
#     trainloader = torch.utils.data.DataLoader(train_dataset_extent, batch_size=36, shuffle=True, num_workers=0)
#     valloader = torch.utils.data.DataLoader(val_dataset_extent, batch_size=36, shuffle=True, num_workers=0)
#     writer = SummaryWriter(comment=f'LR_{0.001}_BS_{255}', log_dir="writ")
#     global_step = 0
#     print("Loading Finished.")
#     with torch.autograd.set_detect_anomaly(True):
#         train_epoch_best_loss = 100.0
#         for epoch in range(500):
#             loss_total = 0
#             for step, data in enumerate(trainloader, 0):
#                 image, seg = data
#                 image = image.to(torch.float32).cuda()
#                 seg = seg.to(torch.int64).cuda()
#                 # seg = seg.cuda()
#
#                 optimizer.zero_grad()
#                 out_seg = model(image)
#
#                 loss_seg = loss_fn(out_seg, seg)
#
#                 # x_vgg = model_vgg(out_seg)
#                 # y_vgg = model_vgg(seg)
#                 # loss_vgg = loss_fn(x_vgg, y_vgg)
#                 loss = loss_seg
#                 loss.backward()
#                 loss_total += loss.item()
#
#                 optimizer.step()
#                 if step % 20 == 0:
#                     print('Current epoch-step: {}-{}  '
#                           '<<<Loss>>>: {}  '
#                           'AllocMem (Mb): {}'.
#                           format(epoch, step,  loss, torch.cuda.memory_allocated() / 1024 / 1024))
#                     print("Detail Loss:<<SEG>>:{}:".
#                           format(loss_seg))
#                     print()
#             loss_total = loss_total/ len(trainloader)
#             writer.add_scalars('train/test', {'train_loss': loss_total}, global_step)
#             scheduler.step()
#     writer.close()


def training_new():
    writer = sw(r'D:\CropSegmentation\tensorboard', filename_suffix='0905')
    ptn_name = "resnet0907_gf2_{}.pth"
    batch_size = 8
    epoch_val = 3  # 验证用的epoch周期
    es_epoch = 10  # 模型早停指标，若连续es_epoch个周期模型未超越最优精度，则停止优化

    model = UNet_Bou().cuda()
    loss_fn = LovaszSoftmaxV1()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Loading 以10%的比例，随机划分训练集和验证集
    trainloader = DataLoader(dataset_tra, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)

    # 定义验证用的metrics函数和其再tensorboard中显示的名字
    metrics_fs = [util_mod.metric_f1, util_mod.metric_ac, util_mod.metric_mse,
                      util_mod.metric_bce, util_mod.metric_tani]
    metrics_names = ['F1', 'AC', 'MSE', 'BCE', 'Tani']

    # 初始化最佳指标
    metrics_best = 0  # 会将val集验证的metrics的第一个metric当成model saving指标
    es_epoch_i = 0  # 初始化早停指标。指标为，

    for epoch in range(200):
        # Training
        model.train()
        for step, data in enumerate(trainloader, 0):
            image, mask = data
            image = image[:, :3, :, :]
            image = image.to(torch.float32).cuda()
            mask = mask.to(torch.int64).cuda()

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
            dataset_metric_recoder(model, trainloader, writer,
                                   "tra", metrics_fs, metrics_names, epoch)
            val_metrics = dataset_metric_recoder(model, valloader, writer,
                                   "val", metrics_fs, metrics_names, epoch)

            # 检验最优参数指标，若为最优参数，则保存模型参数
            if val_metrics[0] > metrics_best:
                torch.save(model.state_dict(), os.path.join(ROOT_PTN, ptn_name.format(val_metrics[0])))
                metrics_best = val_metrics[0]
                es_epoch_i = 0

            # Early Stop
            es_epoch_i += 1
            if es_epoch_i == es_epoch:
                print("Early Stop Happen. Current Epoch is: ", epoch)
                break


if __name__ == "__main__":
    training_new()
