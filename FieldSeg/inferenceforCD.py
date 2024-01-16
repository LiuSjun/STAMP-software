
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
from model.resnet_unet_0903 import UNet_Bou as LSJUNet
import dataset.util as util_dts
import model.util as util_mod
from model.hrnet_mmseg import HRNet
from model.segform_mmseg import SegNet
from model.pspnet_mmseg import PSPNet

ROOT_PTN = r"F:\Code\CD"
from dataset.datasets.data_change_test import DatasetChange

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

    size = 512
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
import cv2
import torch.nn as nn
def predict_image_center( Model, name):

    model = Model.cuda()
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint)
    model.eval()
    batch_size = 2
    testloader = DataLoader(DatasetChange(), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    for step, data in enumerate(testloader, 0):
        image, mask, maskr = data
        image = image
        image = image.to(torch.float32).cuda()

        logit = model.forward_dummy(image)
        seg_pred = nn.Sigmoid()(logit)
        for seg_pred1, mask1, mask2 in zip(seg_pred, mask, maskr):
            seg_pred1 = seg_pred1.cpu().detach().numpy().astype(np.float32)[0]
            UNIT.numpy2img(mask1, seg_pred1)
            seg_pred1[seg_pred1 <= 0.2] = 0
            seg_pred1[seg_pred1 > 0.2] = 1
            _, img_seg = cv2.connectedComponents(seg_pred1.astype(np.uint8))
            UNIT.numpy2img(mask2, img_seg)
            print(mask1)


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

    name_ptn = os.path.join(ROOT_PTN, "hrnet4_0.753479295816177.pth")

    predict_image_center(HRNet(), name_ptn)


