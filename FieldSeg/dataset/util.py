import os, torch, UNIT
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"  # Anaconda 的包和Pytorch的包冲突，在Figure（）时出错


def augmentation(image, mask):
    return image, mask


def preprocessing(image, mask):
    return image, mask


def normalize_parameters(img):
    '''
    获取影像各个波段的normalize parameters
    '''
    img = np.where(img == 0, np.nan, img)
    top = np.nanpercentile(img, 98, axis=(1, 2))
    bottom = np.nanpercentile(img, 2, axis=(1, 2))
    print("影像的参数为：")
    print(top,bottom)
    return top, bottom

def normalize_apply(img, paras):
    para_top, para_bottom = paras
    for i in range(len(para_top)):
        img_bnd = img[i, :, :]
        top, bottom = para_top[i], para_bottom[i]

        img_bnd[img_bnd > top] = top
        img_bnd[img_bnd < bottom] = bottom
        img_bnd = (img_bnd - bottom) / (top - bottom) * 255
        img[i, :, :] = img_bnd
    img = img.astype("uint8")
    return img


def visual3(img, mask, output):
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, (2, 1, 0)]
    mask = mask[0, :, :]
    output = output[0, :, :]

    plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(output)
    plt.show()


def visual2(img, mask):
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, (2, 1, 0)]
    if len(mask.shape) == 3:
        mask = mask[0, :, :]

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()
