from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util
from tqdm import tqdm
import cv2

PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\CD\Processed"
PATH_IMAGES_SPEC = [os.path.join(PATH_IMAGE_ROOT, "ChenduSubRaster.tif")]
PATH_IMAGES_LABEL = [os.path.join(PATH_IMAGE_ROOT, "ChenDu_Label")]
PATH_OUTPUT = r"G:\FinalsForData\GF2\CD\Training"


def normalize_parameters(img):
    '''
    获取影像各个波段的normalize parameters
    '''
    top = np.percentile(img, 98, axis=(1, 2))
    bottom = np.percentile(img, 2, axis=(1, 2))
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

    imgs_spec_subsets = []
    imgs_label_subsets = []
    imgs_bd_subsets = []

    for i in range(len(PATH_IMAGES_SPEC)):
        img_spe = UNIT.img2numpy(PATH_IMAGES_SPEC[i])
        paras = normalize_parameters(img_spe)
        img_spe = normalize_apply(img_spe, paras)

        img_label = UNIT.img2numpy(PATH_IMAGES_LABEL[i])
        img_label = np.where(img_label == 0, 0, 1)
        img_bd = get_boundary(img_label)

        imgs_spec_subsets += subset_dts(img_spe, CHIP_SIZE, 100, 64)
        imgs_label_subsets += subset_dts(img_label, CHIP_SIZE, 100, 64)
        imgs_bd_subsets += subset_dts(img_bd, CHIP_SIZE, 100, 64)

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


def preprocessing(image, mask):
    image = image / 255
    return image, mask


class DatasetCD(Dataset):
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
            images_dir=os.path.join(PATH_OUTPUT, "Spectral"),
            masks_dir=os.path.join(PATH_OUTPUT, "Label"),
            augmentation=dataset.util.augmentation,
            preprocessing=preprocessing,
    ):
        self.images_names = os.listdir(images_dir)
        self.masks_names = self.images_names

        self.images_fps = [os.path.join(images_dir, name) for name in self.images_names]
        self.masks_fps = [os.path.join(masks_dir, name) for name in self.masks_names]

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


if __name__ == "__main__":
    '''
    运行整个程序前，先运行prepare，对Anhui数据进行分块处理。
    修改PATH_IMAGE_ROOT、PATH_IMAGES_SPEC、PATH_LABEL_IMAGES和PATH_OUTPUT以匹配用户自己的工作路径
    '''
    prepare()
    pass
