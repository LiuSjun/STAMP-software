'''
This file is the input program of crop segmentation network.
Author: Shuaijun Liu
Date: 2022/7/4
Email: 18202739183@qq.com
'''
from torch.utils.data import DataLoader, random_split
import numpy as np
import os, torch, UNIT
import torch.optim as optim
from tqdm import tqdm
from model.resnet_unet_0714 import UNet as BasicUNet
import dataset.util as util_dts
import model.util as util_mod

from tensorboardX import SummaryWriter as sw

ROOT_PTN = r"E:\Newsegdataset\崇州\pths"
ROOT_RST = r"E:\Newsegdataset\崇州\result"

###### IMPORT Training Data
from dataset.dataset_INT_aug import DatasetGF2
from dataset.datasets.dataset_ah import DatasetAH
from dataset.datasets.dataset_bs import DatasetBS
from dataset.datasets.dataset_SZ import DatasetSZ
from dataset.datasets.dataset_ZN import DatasetZN
from dataset.datasets.dataset_xzlable import DatasetXZ
from dataset.datasets.dataset_J import DatasetJ
from dataset.datasets.dataset_D import DatasetD
from dataset.datasets.dataset_lz import DatasetLZ
from dataset.datasets.dataset_gs import DatasetGS
from dataset.datasets.dataset_hlj import DatasetHLJ
from dataset.datasets.dataset_ms import DatasetMS
from dataset.datasets.dataset_zj import DatasetZJ
from dataset.datasets.dataset_zz import DatasetZZ
from dataset.datasets.dataset_test import DatasetT

datasets = [DatasetAH(),DatasetSZ()]#, DatasetBS(), DatasetXZ(), DatasetJ(), DatasetD()] ##, DatasetSZ(), DatasetZN(), DatasetBS()] ##,DatasetCD(), DatasetGS() DatasetHLJ(), DatasetMS(), DatasetZJ(), DatasetZZ()]
dataset = DatasetGF2(
    datasets
)

dataset_tra, dataset_val = dataset.dataset_split(0.9)
# dataset_tra = dataset


def dataset_metric_recoder(model, dataset, writer: sw, writer_name, metrics, metrics_names, global_step):
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
    writer.add_scalars(writer_name, dict(zip(metrics_names, arr_metrics)), global_step)

    print("Epoch {}: ".format(global_step), end=' ')
    for i in range(len(metrics)):
        print(metrics_names[i], ":", arr_metrics[i], end='; ')
    print()

    return arr_metrics


def tranining():
    # dataset = DatasetGF2(
    #     (dts_ah.DatasetAH(), dts_cd.DatasetCD(), dts_zj.DatasetZJ(), dts_hlj.DatasetHLJ(), )
    # )
    writer = sw(r'E:\Newsegdataset\崇州\tensorboard\base', filename_suffix='0905')
    ptn_name = "resnet_seg_{}.pth"
    batch_size = 40

    epoch_val = 1  # 验证用的epoch周期
    es_epoch = 20  # 模型早停指标，若连续es_epoch个周期模型未超越最优精度，则停止优化

    model = BasicUNet().cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_name = ''  # r'E:\Newsegdataset\xizang\pths\resnet0905_gf2_0.9095530907313029.pth'
    if load_name:
        model.load_state_dict(
            torch.load(load_name, map_location=device)
        )

    loss_fn = util_mod.TanimotoLoss()  # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

    # Loading 以10%的比例，随机划分训练集和验证集
    trainloader = DataLoader(dataset_tra, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0) ##DatasetT(), batch_size=batch_size, shuffle=True, num_workers=0)

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
            dataset_metric_recoder(model, trainloader, writer,
                                   "tra", metrics_fs, metrics_names, epoch)
            dataset_metric_recoder(model, valloader, writer,
                                   "val", metrics_fs, metrics_names, epoch)
            val_metrics = dataset_metric_recoder(model, testloader, writer,
                                   "test", metrics_fs, metrics_names, epoch)

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
    tranining()

