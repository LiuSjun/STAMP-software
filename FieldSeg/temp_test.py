'''
This file is the input program of crop segmentation network.
Author: Licong Liu
Date: 2022/7/4
Email: 543476459@qq.com
'''
from torch.utils.data import DataLoader, random_split
import numpy as np
import os, torch, UNIT
import torch.optim as optim
from tqdm import tqdm
# from dataset.dataset_anhui import DatasetAH
from dataset_dynamic.dataset_anhui_dynamic import DatasetAH
from dataset.dataset_INT import DatasetGF2
# from model import unet
from model.resnet_unet_0714 import UNet as BasicUNet
from model.resnet_unet_rgb import UNet as RGBUNet
from model.resnet_unet_bou_grad import UNet_Bou as NIRGRAD  #关于边界的模型
from model.resnet_unet_0903 import UNet_Bou as LSJUNet
import dataset.util as util_dts
import model.util as util_mod
from cropfield import remove_noise, mark2rgb
import cv2
#
# ROOT_PTN = r"E:\Newsegdataset\sichuan\pths"
# ROOT_RST = r"E:\Newsegdataset\sichuan\result"    #"E:\Newsegdataset\xizang\result\6dataset"
ROOT_PTN = r"E:\Newsegdataset\dazhuang\pths"
ROOT_RST = r"G:\甘肃苹果\91下载\阳川大庄20190712"  #r"E:\Newsegdataset\dazhuang\result11"

def mat_xy_pad(img, x, y, width=2):
    img[0, 0, :width, :] = 0
    img[0, 0, x - width:, :] = 0
    img[0, 0, :, :width] = 0
    img[0, 0, :, y - width:] = 0
    return img


def predict_image(img, Model, name, interval=16, start=16, mask=None):
    '''
    将整个影像分割为各个255 * 255的小块，推测完后加回去。
    '''

    model = Model.cuda()
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint)
    model.eval()

    size = 2048
    start = 400
    interval = 800
    b, xlen, ylen = img.shape
    half_size = int(size / 2)
    x_center = np.arange(start, xlen, interval, dtype=int)
    y_center = np.arange(start, ylen, interval, dtype=int)
    x_center, y_center = np.meshgrid(x_center, y_center)

    img_count = np.zeros((xlen, ylen), dtype=np.int16)
    img_label = np.zeros((xlen, ylen), dtype=np.float32)

    xlen_chip, ylen_chip = x_center.shape
    with torch.autograd.set_detect_anomaly(True) and torch.no_grad():
        for i in tqdm(range(xlen_chip)):
            for j in range(ylen_chip):
                xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min((x_center[i, j] + half_size, xlen))
                yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min((y_center[i, j] + half_size, ylen))
                subset_img = np.zeros((b, size, size), dtype=np.float32)
                xsize, ysize = xloc1 - xloc0, yloc1 - yloc0
                subset_img[:, :xsize, :ysize] = img[:, xloc0:xloc1, yloc0:yloc1]
                if np.max(subset_img) == 0:
                    continue
                no_value_loc = np.where(np.mean(subset_img, axis=0) == 0, 0, 1)

                subset_img = np.expand_dims(subset_img, 0)
                subset_img_torch = torch.from_numpy(subset_img).cuda()

                img_sub_label = model(subset_img_torch)  # critical
                img_sub_label = img_sub_label.cpu().numpy().astype(float)
                img_sub_label *= no_value_loc

                if mask is not None:
                    img_count[xloc0:xloc1, yloc0:yloc1] += mask[:xsize, :ysize]
                    img_label *= mask
                else:
                    img_count[xloc0:xloc1, yloc0:yloc1] += 1
                img_label[xloc0:xloc1, yloc0:yloc1] += img_sub_label[0, 0, :xsize, :ysize]
    epsilon = 1e-7
    img_label = img_label / (img_count + epsilon)
    return img_label


def predict_image_center(img, Model, name):
    '''
    将整个影像分割为各个1024 * 1024的小块，推测完后加回去。
    注意，具体的裁剪大小和影像的剩余长宽有关，例如，很可能裁剪除1024 * 120的小图像。
    若任意一边长小于100，则不进行该小图像的推断。
    该代码为删除边界后的推测代码，影像进行24 pixel的Padding后，推测回去。
    【这意味着影像边缘的24个pixel无法被预测？】
    【先看看效果吧】
    '''
    size = 2048
    start = 400
    interval = 800
    padding = 100

    model = Model.cuda()
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint)
    model.eval()

    b, xlen, ylen = img.shape
    half_size = int(size / 2)
    x_center = np.arange(start, xlen, interval, dtype=int)
    y_center = np.arange(start, ylen, interval, dtype=int)
    x_center, y_center = np.meshgrid(x_center, y_center)

    img_count = np.zeros((xlen, ylen), dtype=np.int16)  # 记录每个像素被预测的次数
    img_label = np.zeros((xlen, ylen), dtype=np.float16)  # 记录每次预测时，该像素所代表的概率

    xlen_chip, ylen_chip = x_center.shape
    with torch.autograd.set_detect_anomaly(True) and torch.no_grad():
        for i in tqdm(range(xlen_chip)):
            for j in range(ylen_chip):
                # 获取到预测图像在原始图像上的坐标
                xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min((x_center[i, j] + half_size, xlen))
                yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min((y_center[i, j] + half_size, ylen))
                xsize, ysize = xloc1 - xloc0, yloc1 - yloc0
                # Using 512 * 512 image to predict
                subset_img = np.zeros((b, size, size), dtype=np.float32)
                # image used to inference
                subset_img[:, :xsize, :ysize] = img[:, xloc0:xloc1, yloc0:yloc1].astype(np.float32)
                if np.max(subset_img) == 0:
                    continue
                zero_mask = np.where(np.max(subset_img, axis=0) == 0, 0, 1)

                subset_img = np.expand_dims(subset_img, 0)
                subset_img_torch = torch.from_numpy(subset_img).cuda()

                img_sub_label = model(subset_img_torch)  # critical
                img_sub_label = img_sub_label.cpu().numpy().astype(np.float16)
                img_sub_label *= zero_mask  # mask the result using zero value area

                img_sub_count = np.ones(img_sub_label.shape)
                if xloc0 != 0 and yloc0 != 0 and xloc1 != xlen and yloc1 != ylen:
                    img_sub_count = mat_xy_pad(img_sub_count, xsize, ysize, width=padding).astype(int)
                    img_sub_label = mat_xy_pad(img_sub_label, xsize, ysize, width=padding)

                img_label[xloc0:xloc1, yloc0:yloc1] += img_sub_label[0, 0, :xsize, :ysize].astype(np.float16)
                img_count[xloc0:xloc1, yloc0:yloc1] += img_sub_count[0, 0, :xsize, :ysize].astype(np.int16)

    # epsilon = 1e-7
    # img_label = img_label / (img_count + epsilon)
    epsilon = float(1e-7)
    img_label = img_label.astype("float16")
    img_count = (img_count + epsilon)
    img_count = img_count.astype("float16")
    img_label = img_label / img_count
    img_label = img_label
    return img_label.astype("float32")


def predict_subimg():
    dataset = DatasetGF2()

    # model = unet.UNet(4).cuda()
    model = BasicUNet().cuda()
    name_ptn = os.path.join(ROOT_PTN, "model_state_e190.pth")
    checkpoint = torch.load(name_ptn)
    model.load_state_dict(checkpoint)

    print("Data Loading...")
    trainloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    print("Loading Finished.")
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(200):
            for step, data in enumerate(trainloader, 0):
                image, mask = data
                image = image.to(torch.float32).cuda()
                mask = mask.to(torch.float32).cuda()
                outputs = model(image)

                with torch.no_grad():
                    outputs = outputs.cpu().numpy().astype(float)
                    image = image.cpu().numpy().astype(float)
                    mask = mask.cpu().numpy().astype(float)
                    util_dts.visual3(image[0, :, :, :], mask[0, :, :, :], outputs[0, :, :, :])


if __name__ == "__main__":
    # LSJ 不要除255 + eval 结果最好
    # LSJ 不要除255 会出现重影，证明，重影的现线确实是eval导致的
    # LLC 必须要除以255，因为数据本身就是除了255的
    # predict_subimg()
    img_target_path = r"D:\DeepLearn\SAM\邛崃样本"    #r"F:\QGIS download\mk0213-2"
    image_list = os.listdir(img_target_path)
    for k in range(len(image_list)):
        if image_list[k][-4:] == '.dat':
            # img_target = img_target_path + image_list[k]
            img_target = os.path.join(img_target_path, image_list[k])
            # img_target = r"D:\CropSegmentation\data\DynamicCrop\区域2_GS"
            img_target, proj, geot = UNIT.img2numpy(img_target, geoinfo=True)

            img_target = img_target.astype("int8")
            img_target[img_target != 1] = 0
            # img_target = np.nan_to_num(img_target, nan=0)

            UNIT.numpy2img(os.path.join(img_target_path, "{}.tif".format(image_list[k][:-4])), img_target, proj=proj, geot=geot)  #
            del img_target   #, rgb


