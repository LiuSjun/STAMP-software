'''
This file is the input program of crop segmentation network.
该函数只用RGB进行训练
Author: Licong Liu
Date: 2022/9/6
Email: 543476459@qq.com
'''
from torch.utils.data import DataLoader, random_split
import numpy as np
import os, torch, UNIT
import torch.optim as optim
from model.resnet_unet_0714 import UNet
import model.util as util_mod
from torch.utils.tensorboard import SummaryWriter as sw


ROOT_PTN = r"D:\CropSegmentation\pths"
ROOT_RST = r"D:\CropSegmentation\result"

###### IMPORT Training Data
from dataset.dataset_INT_dw import DatasetGF2
from dataset.dataset_dw.dataset_base import DatasetBase

datasets = (DatasetBase(r"D:\CropSegmentation\data\GF2\AH\Training"),
            DatasetBase(r"D:\CropSegmentation\data\GF2\GS\Training"),
            DatasetBase(r"D:\CropSegmentation\data\GF2\HLJ\Training"))
dataset = DatasetGF2(
    datasets
)
dataset_tra, dataset_val = dataset.dataset_split(0.9)
dataset_tst = DatasetGF2((DatasetBase(r"D:\CropSegmentation\data\GF2\CD\Training"), ))


def dataset_metric_recoder(model, dataset, writer: sw, writer_name, metrics, metrics_names, global_step):
    arr_metrics = np.empty((0, len(metrics)))
    with torch.no_grad():
        for step, data in enumerate(dataset, 0):
            image, mask, dw = data
            image[:, 3, :, :] = 0  # dw[:, 0, :, :]
            image = image / 255
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
    writer = sw(r'D:\CropSegmentation\tensorboard\RGB_UNET_NO_DW', filename_suffix='0910')
    ptn_name = "resnet0910_UNET_DW_{}.pth"
    batch_size = 8
    epoch_val = 1  # 验证用的epoch周期
    es_epoch = 10  # 模型早停指标，若连续es_epoch个周期模型未超越最优精度，则停止优化

    model = UNet().cuda()
    loss_fn = util_mod.TanimotoLoss()  # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Loading 以10%的比例，随机划分训练集和验证集
    trainloader = DataLoader(dataset_tra, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset_tst, batch_size=batch_size, shuffle=True, num_workers=0)

    # 定义验证用的metrics函数和其再tensorboard中显示的名字
    metrics_fs = [util_mod.metric_f1, util_mod.metric_ac, util_mod.metric_mse,
                      util_mod.metric_bce, util_mod.metric_tani]
    metrics_names = ['F1', 'AC', 'MSE', 'BCE', 'Tani']

    # 初始化最佳指标
    metrics_best = 0  # 会将val集验证的metrics的第一个metric当成model saving指标
    es_epoch_i = 0  # 初始化早停指标。指标为，

    for epoch in range(100):
        # Training
        model.train()
        for step, data in enumerate(trainloader, 0):
            image, mask, dw = data
            image[:, 3, :, :] = 0 # dw[:, 0, :, :]
            image = image / 255
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

            val_metrics = dataset_metric_recoder(model, valloader, writer,
                                   "val", metrics_fs, metrics_names, epoch)

            dataset_metric_recoder(model, testloader, writer,
                                   "test", metrics_fs, metrics_names, epoch)

            '''
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
            '''
        print("LR:", scheduler.get_last_lr())


if __name__ == "__main__":
    tranining()

