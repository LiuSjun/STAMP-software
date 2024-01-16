'''
GID 数据集的格式和其他的Dataset不一样
他的保存路径为影像名之下【因为影像太多了】
同时，该数据集还需要进行波段反转的预处理

该数据集的特点
没有地理坐标信息
以Nir R G B 的顺序存储波段
'''

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util, re
from tqdm import tqdm
import cv2
from osgeo import gdal


os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GID Dataset\Raw"
PATH_OUTPUT = r"D:\CropSegmentation\data\GID Dataset\Training"


PATH_IMAGES_SPEC = os.path.join(PATH_IMAGE_ROOT, "Spectral")
PATH_IMAGES_LABEL = os.path.join(PATH_IMAGE_ROOT, "label")
PATH_OUTPUT_ROOT_Lable = os.path.join(PATH_OUTPUT, "Label")
PATH_OUTPUT_ROOT_Spectral = os.path.join(PATH_OUTPUT, "Spectral")

# check the integrity of raw gid images
# return the file paths


def visual(img, mask):
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, (2, 1, 0)] / 255
    mask = mask[0, :, :]

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()


def files_check():
    path_images_specs = os.listdir(PATH_IMAGES_SPEC)
    path_files = []
    for i in tqdm(range(len(path_images_specs))):
        # Obtain file names
        path_image_spec = path_images_specs[i]
        path_image_label = path_image_spec.split('.')[0] + "_label.tif"

        # check whether the file is tif
        if path_image_spec[-4:] != ".tif":
            continue
        print(path_image_spec)
        dts_image = gdal.Open(os.path.join(PATH_IMAGES_SPEC, path_image_spec))
        dts_label = gdal.Open(os.path.join(PATH_IMAGES_LABEL, path_image_label))
        if dts_image is None:
            print(path_image_spec)
            continue
        if dts_label is None:
            print(path_image_label)
            continue

        path_files.append((os.path.join(PATH_IMAGES_SPEC, path_image_spec),
                           os.path.join(PATH_IMAGES_LABEL, path_image_label)))

    return path_files


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


def prepare():
    '''
    Chip the whole spectral and label image to subset spectral images 和 crop.label and boundary label.
    '''
    CHIP_SIZE = 256

    filenames = files_check()
    for n, filename in enumerate(filenames):
        name_spec, name_label = filename
        prefix = re.split('-|__', name_spec)[-2]

        path_output_root_spec = os.path.join(PATH_OUTPUT_ROOT_Spectral, prefix)
        path_output_root_label = os.path.join(PATH_OUTPUT_ROOT_Lable, prefix)

        if not os.path.exists(path_output_root_spec):
            os.mkdir(path_output_root_spec)
        if not os.path.exists(path_output_root_label):
            os.mkdir(path_output_root_label)

        img_spe = UNIT.img2numpy(name_spec)
        img_label = UNIT.img2numpy(name_label)

        if img_spe is None:
            print("ERROR!!! - ", img_spe)
            continue
        if img_label is None:
            print("ERROR!!! - ", img_label)
            continue
        img_label = img_label.astype("int32")
        img_label = img_label[0, :, :] * 1000000 + img_label[1, :, :] * 1000 + img_label[2, :, :]
        img_label = np.where(img_label == 255000, 1, 0)
        img_label = img_label.astype("uint8")

        imgs_spec_subsets = subset_dts(img_spe, CHIP_SIZE, 100, 200)
        imgs_label_subsets = subset_dts(img_label, CHIP_SIZE, 100, 200)
        len_imgs = len(imgs_spec_subsets)
        for i in range(len_imgs):
            UNIT.numpy2img(os.path.join(path_output_root_spec, "{}.tif".format(i)), imgs_spec_subsets[i][::-1, :, :])
            UNIT.numpy2img(os.path.join(path_output_root_label, "{}.tif".format(i)), imgs_label_subsets[i])
        print("{} Finished, Name - {}".format(n, prefix))


class DatasetGID(Dataset):
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    CLASSES = ['CropLand', 'Others']
    CLASSES_VALUE = [255, 100, 0]

    def __init__(
            self,
            augmentation=dataset.util.augmentation,
            preprocessing=dataset.util.preprocessing,
    ):
        self.img_dirs = os.listdir(PATH_OUTPUT_ROOT_Spectral)
        path_imgs = []
        path_labels = []
        for dir in self.img_dirs:
            names = os.listdir(os.path.join(PATH_OUTPUT_ROOT_Spectral, dir))
            for name in names:
                if name[-4:] == ".tif":
                    path_imgs.append(os.path.join(PATH_OUTPUT_ROOT_Spectral, dir, name))
                    path_labels.append(os.path.join(PATH_OUTPUT_ROOT_Lable, dir, name))

        self.images_fps = path_imgs
        self.masks_fps = path_labels
        # convert str names to class values on masks

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = UNIT.img2numpy(self.images_fps[i]).astype(np.float32)
        mask = UNIT.img2numpy(self.masks_fps[i]).astype(np.uint8)
        mask = np.where(mask == 0, 0, 1).astype(np.uint8)
        mask = np.expand_dims(mask, axis=0)

        # visualize(image=image, mask=mask)
        # apply augmentations
        if self.augmentation:
            image, mask = self.augmentation(image=image, mask=mask)

        # apply preprocessing
        if self.preprocessing:
            image, mask = self.preprocessing(image=image, mask=mask)
        return image, mask

    def __len__(self):
        return len(self.images_fps)


def dataset_test():
    '''
    遍历Dataset中所有的影像
    '''
    iter = 0
    dts_gid = DatasetGID()
    for image, mask in tqdm(dts_gid):
        visual(image, mask)
        iter += 1
    print(iter)


if __name__ == "__main__":
    # path_files = files_check()
    # print(path_files)

    # prepare()
    dataset_test()

    # name_label = "Z:\\Crop-Competition\\GF-Data(GID)\\label_5classes\\GF2_PMS1__L1A0000564539-MSS1_label.tif"
    # img_label = UNIT.img2numpy(name_label)
    pass