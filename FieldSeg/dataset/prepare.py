'''
该函数用于利用spectral.tif和label.tif生成spec、extend和label文件
文件为0-255格式的，BGR格式

该函数用于裁剪对应区域的训练数据

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
    y_center = np.arange(start, ylen, interval, dtype=int)
    x_center, y_center = np.meshgrid(x_center, y_center)

    xlen_chip, ylen_chip = x_center.shape
    img_list = []
    for i in tqdm(range(xlen_chip)):
        for j in range(ylen_chip):
            xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min((x_center[i, j] + half_size, xlen))
            yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min((y_center[i, j] + half_size, ylen))
            subset_img = np.zeros((b, size, size), dtype=img.dtype)
            # print(subset_img.shape)
            subset_img[:, 0:xloc1 - xloc0, 0:yloc1 - yloc0] = img[:, xloc0:xloc1, yloc0:yloc1]
            img_list.append(subset_img)
    return img_list


def get_boundary(label, kernel_size=(3, 3)):    #(3, 3)
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel, 0, 1)
    tlabel = cv2.dilate(
        temp,
        cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            kernel_size),
        iterations=1
        )
    # while True:
    #     open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, tlabel)
    #     temp = cv2.subtract(binary, open)
    #     eroded = cv2.erode(binary, tlabel)
    #     skel = cv2.bitwise_or(skel, temp)
    #     binary = eroded.copy()
    #
    #     if cv2.countNonZero(binary) == 0:
    #         break
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel


def prepare(interval=200):
    '''
    Chip the whole spectral and label image to subset spectral images 和 crop.label and boundary label.
    '''
    CHIP_SIZE = 512

    imgs_spec_subsets = []
    imgs_label_subsets = []
    imgs_bd_subsets = []

    for i in range(len(PATH_IMAGES_SPEC)):

        img_spe = UNIT.img2numpy(PATH_IMAGES_SPEC[i])
        paras = dataset.util.normalize_parameters(img_spe)
        img_spe = dataset.util.normalize_apply(img_spe, paras)

        img_label = UNIT.img2numpy(PATH_IMAGES_LABEL[i])
        img_label = np.where(img_label == 0, 0, 1)    ######################
        print(img_label.shape)
        # img_label = np.where(img_label == 65535, 0, 1)   ##图斑有值，但其余区域为空值时使用。
        # img_label = np.nan_to_num(img_label, nan=0)
        img_bd = get_boundary(img_label)
        imgs_spec_subsets += subset_dts(img_spe, CHIP_SIZE, 100, interval)
        imgs_label_subsets += subset_dts(img_label, CHIP_SIZE, 100, interval)
        imgs_bd_subsets += subset_dts(img_bd, CHIP_SIZE, 100, interval)

    if not os.path.exists(os.path.join(PATH_OUTPUT, "Spectral")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Spectral"))
    if not os.path.exists(os.path.join(PATH_OUTPUT, "Label")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Label"))
    if not os.path.exists(os.path.join(PATH_OUTPUT, "Boundary")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Boundary"))

    len_imgs = len(imgs_spec_subsets)
    for i in range(len_imgs):
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Spectral", "{}.tif".format(i)), imgs_spec_subsets[i])
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Label", "{}.tif".format(i)), imgs_label_subsets[i])
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Boundary", "{}.tif".format(i)), imgs_bd_subsets[i])

def prepareG1(interval=400):
    '''
    Chip the whole spectral and label image to subset spectral images 和 crop.label and boundary label.
    '''
    CHIP_SIZE = 512

    imgs_spec_subsets = []
    imgs_label_subsets = []
    imgs_bd_subsets = []

    for i in range(len(PATH_IMAGES_SPEC)):
        img_spe = UNIT.img2numpy(PATH_IMAGES_SPEC[i])[1:]
        paras = dataset.util.normalize_parameters(img_spe)
        img_spe = dataset.util.normalize_apply(img_spe, paras)

        img_label = UNIT.img2numpy(PATH_IMAGES_LABEL[i])[0]
        img_label = np.where(img_label == 255, 0, 1)
        img_bd = get_boundary(img_label)

        imgs_spec_subsets += subset_dts(img_spe, CHIP_SIZE, 100, interval)
        imgs_label_subsets += subset_dts(img_label, CHIP_SIZE, 100, interval)
        imgs_bd_subsets += subset_dts(img_bd, CHIP_SIZE, 100, interval)

    if not os.path.exists(os.path.join(PATH_OUTPUT, "Spectral")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Spectral"))
    if not os.path.exists(os.path.join(PATH_OUTPUT, "Label")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Label"))
    if not os.path.exists(os.path.join(PATH_OUTPUT, "Boundary")):
        os.mkdir(os.path.join(PATH_OUTPUT, "Boundary"))

    len_imgs = len(imgs_spec_subsets)
    for i in range(len_imgs):
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Spectral", "{}.tif".format(i)), imgs_spec_subsets[i])
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Label", "{}.tif".format(i)), imgs_label_subsets[i])
        UNIT.numpy2img(os.path.join(PATH_OUTPUT, "Boundary", "{}.tif".format(i)), imgs_bd_subsets[i])


if __name__ == "__main__":

    # Area Anhui
    PATH_IMAGE_ROOT = r"G:\崇州影像\Processed\数据集标签"  ##"E:\Newsegdataset\dazhuang\Process20190721\Processed"   #"D:\项目执行\Processed\Processed\新建文件夹"
    os.chdir(PATH_IMAGE_ROOT)
    PATH_IMAGES_SPEC = ["Subset{}.tif".format(i) for i in range(8,9)]
    PATH_IMAGES_LABEL = ["Subset{}_object.dat".format(i) for i in range(8,9)]
    PATH_OUTPUT = r"G:\崇州影像\Processed\数据集标签\Training123"  ##"E:\Newsegdataset\dazhuang\Process20190721\Training12"  #"D:\项目执行\Processed\Training"
    if not os.path.exists(PATH_OUTPUT):
        os.mkdir(PATH_OUTPUT)
    prepare()

    # ## Area ChengDu
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\CD\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["ChenduSubRaster.tif"]
    # PATH_IMAGES_LABEL = ["ChenDu_Label"]
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\CD\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()
    #
    # ## Area Hei LongJiang
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\HLJ\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["GF2_20170731_Subset0.dat", "GF2_20170731_Subset1.dat"]
    # PATH_IMAGES_LABEL = ["sub0label", "sub1label"]
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\HLJ\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()
    #
    # ## Area Ze Jiang
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\ZJ\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["Sub{}.tif".format(i) for i in range(5)]
    # PATH_IMAGES_LABEL = ["Sub0_Area.tif", "Sub1_Area.dat", "Sub2_label.dat", "Sub3_Area", "Sub4_Area"]
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\ZJ\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()

    # ## Area MS
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\MS\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["meishan_fusionNIR.hdr.dat.enp.dat", "sub2fusionresult.dat"]
    # PATH_IMAGES_LABEL = ["sub1label.tif", "sub2label.tif"]
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\MS\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()
    #
    # ## Area GS
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\GS\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["sub1fusion.dat", "sub2fusion.dat"]
    # PATH_IMAGES_LABEL = ["sub1label.tif", "sub2label.tif"]
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\GS\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()
    #
    # ## Area ZZ
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\ZZ\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["sub1fusion.dat", "sub2fusion.dat"]
    # PATH_IMAGES_LABEL = ["sub1label.tif", "sub2label.tif"]
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\ZZ\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()

    # ## Area G1 NIR-R-G-B
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\G_Label_RGB\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = os.listdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_LABEL = PATH_IMAGES_SPEC
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\G_Label_RGB\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepareG1()

    ## Area G-HB-SD
    # PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\G_HB_SD\Processed"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["HT_s1.dat", "HT_s2.tif", "ZJ_s1.dat", "ZJ_s2.dat", "ZJ_s3.dat", "ZJ_s4.dat", "ZJ_s5.dat", "ZJ_s6.dat"]
    # PATH_IMAGES_LABEL = ["HT_s1label", "HT_s2lable", "ZJ_s1label", "ZJ_s2label", "ZJ_s3label", "ZJ_s4label", "ZJ_s5label", "ZJ_s6label"]
    # PATH_OUTPUT = r"G:\FinalsForData\GF2\G_HB_SD\Training"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()



    # # Test
    # PATH_IMAGE_ROOT = r"Z:\Crop-Competition\决赛数据\TrainingData\区域2"
    # os.chdir(PATH_IMAGE_ROOT)
    # PATH_IMAGES_SPEC = ["区域2_GS_Training.dat"]
    # PATH_IMAGES_LABEL = ["区域2_Label.dat"]
    # PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\Test"
    # if not os.path.exists(PATH_OUTPUT):
    #     os.mkdir(PATH_OUTPUT)
    # prepare()

    print("FINISHED")
