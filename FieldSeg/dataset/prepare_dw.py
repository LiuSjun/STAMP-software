'''
该代码是包含了DW版本的prepare程序
'''
from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util
from tqdm import tqdm
import cv2


def subset_dts(img, size, start, interval):
    if len(img.shape) == 2:
        xlen, ylen = img.shape
        img = img.reshape((1, xlen, ylen))
        b = 1
    else:
        b, xlen, ylen = img.shape
    half_size = int(size / 2)
    x_center = np.arange(start, xlen, interval, dtype=int)
    y_center = np.arange(start, xlen, interval, dtype=int)
    x_center, y_center = np.meshgrid(x_center, y_center)

    xlen_chip, ylen_chip = x_center.shape
    img_list = []
    for i in tqdm(range(xlen_chip)):
        for j in range(ylen_chip):
            xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min((x_center[i, j] + half_size, xlen))
            yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min((y_center[i, j] + half_size, ylen))
            subset_img = np.zeros((b, size, size), dtype=img.dtype)
            subset_img[:, 0:xloc1 - xloc0, 0:yloc1 - yloc0] = img[:, xloc0:xloc1, yloc0:yloc1]
            img_list.append(subset_img)
    return img_list


def get_boundary(label, kernel_size=(3, 3)):
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel, 0, 1)
    tlabel = cv2.dilate(
        temp,
        cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            kernel_size),
        iterations=1
        )
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel


def prepare(interval=400):
    '''
    Chip the whole spectral and label image to subset spectral images 和 crop.label and boundary label.
    '''
    CHIP_SIZE = 512

    imgs_spec_subsets = []
    imgs_label_subsets = []
    imgs_bd_subsets = []
    imgs_dw_subsets = []

    for i in range(len(PATH_IMAGES_SPEC)):
        img_spe = UNIT.img2numpy(PATH_IMAGES_SPEC[i])
        paras = dataset.util.normalize_parameters(img_spe)
        img_spe = dataset.util.normalize_apply(img_spe, paras)

        img_label = UNIT.img2numpy(PATH_IMAGES_LABEL[i])
        img_label = np.where(img_label == 0, 0, 1)
        img_bd = get_boundary(img_label)

        img_dw = UNIT.img2numpy(PATH_DWS[i])

        imgs_spec_subsets += subset_dts(img_spe, CHIP_SIZE, 100, interval)
        imgs_label_subsets += subset_dts(img_label, CHIP_SIZE, 100, interval)
        imgs_bd_subsets += subset_dts(img_bd, CHIP_SIZE, 100, interval)
        imgs_dw_subsets += subset_dts(img_dw, CHIP_SIZE, 100, interval)

    if not os.path.exists(os.path.join(PATH_OUTPUT, "Spectral")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Spectral"))
    if not os.path.exists(os.path.join(PATH_OUTPUT, "Label")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Label"))
    if not os.path.exists(os.path.join(PATH_OUTPUT, "Boundary")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Boundary"))
    if not os.path.exists(os.path.join(PATH_OUTPUT, "DW")):
        os.mkdir(os.path.join(PATH_OUTPUT, "DW"))

    len_imgs = len(imgs_spec_subsets)
    for i in range(len_imgs):
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Spectral", "{}.tif".format(i)), imgs_spec_subsets[i])
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Label", "{}.tif".format(i)), imgs_label_subsets[i])
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Boundary", "{}.tif".format(i)), imgs_bd_subsets[i])
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "DW", "{}.tif".format(i)), imgs_dw_subsets[i])


if __name__ == "__main__":

    # ## Area Anhui
    # PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\AH\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["Subset{}.tif".format(i) for i in range(3)]
    # PATH_IMAGES_LABEL = ["Subset{}_object.dat".format(i) for i in range(3)]
    # PATH_DWS = ["Subset{}_DW.tif".format(i) for i in range(3)]
    # PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\AH\Training"
    # prepare()
    #
    # ## Area ChengDu
    # PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\CD\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["ChenduSubRaster.tif"]
    # PATH_IMAGES_LABEL = ["ChenDu_Label"]
    # PATH_DWS = ["ChenduSubRaster_DW.tif"]
    # PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\CD\Training"
    # prepare()

    ## Area Hei LongJiang
    PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\HLJ\Processed"
    os.chdir(PATH_IMAGE_ROOT)
    PATH_IMAGES_SPEC = ["GF2_20170731_Subset0.dat", "GF2_20170731_Subset1.dat"]
    PATH_IMAGES_LABEL = ["sub0label", "sub1label"]
    PATH_DWS = ["GF2_20170731_Subset0_DW.tif", "GF2_20170731_Subset1_DW.tif"]
    PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\HLJ\Training"
    prepare()
    #
    # ## Area Ze Jiang
    # PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\ZJ\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["Sub{}.tif".format(i) for i in range(5)]
    # PATH_IMAGES_LABEL = ["Sub0_Area.tif", "Sub1_Area.dat", "Sub2_label.dat", "Sub3_Area", "Sub4_Area"]
    # PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\ZJ\Training"
    # prepare()
    #
    # ## Area MS
    # PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\MS"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["sub1ps.tif", "sub11.tif", "sub2ps.tif", "sub21.tif"]
    # PATH_IMAGES_LABEL = ["sub1label.tif", "sub1label.tif", "sub2label.tif", "sub2label.tif"]
    # PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\MS\Training"
    # prepare()
    #
    ## Area GS
    PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\GS\Processed"
    os.chdir(PATH_IMAGE_ROOT)
    PATH_IMAGES_SPEC = ["sub1_20200520.tif", "sub1_20201005.tif", "sub1_20210225.tif", "sub2_20200520.tif", "sub2_20201005.tif", "sub2_20210225.tif"]
    PATH_DWS = ["sub1_DW.tif", "sub1_DW.tif", "sub1_DW.tif", "sub2_DW.tif", "sub2_DW.tif", "sub2_DW.tif"]
    PATH_IMAGES_LABEL = ["sub1label.tif", "sub1label.tif", "sub1label.tif", "sub2label.tif", "sub2label.tif", "sub2label.tif"]
    PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\GS\Training"
    prepare()

    # ## Area ZZ
    # PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\ZZ"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["sub1.tif", "sub2.tif"]
    # PATH_IMAGES_LABEL = ["sub1label.tif", "sub2label.tif"]
    # PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\ZZ\Training"
    # prepare()

    ## Test
    # PATH_IMAGE_ROOT = r"Z:\Crop-Competition\决赛数据\TrainingData\区域2"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["区域2_GS_Training.dat"]
    # PATH_IMAGES_LABEL = ["区域2_Label.dat"]
    # PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\Test"
    # prepare()

    print("FINISHED")
