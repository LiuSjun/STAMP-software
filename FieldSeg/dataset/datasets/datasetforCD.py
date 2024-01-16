from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util
from tqdm import tqdm
import cv2

PATH_OUTPUT = r"G:\ChangeDete\train"


def preprocessing(imageA,imageB, mask):
    imageA = imageA / 255
    imageB = imageB / 255
    return imageA, imageB, mask
from albumentations import (
    RandomRotate90, Flip,  Compose, RandomBrightnessContrast, HueSaturationValue, RGBShift, OneOf, ShiftScaleRotate
)
import cv2
def augmentation(**images):
    '''
    通过图像左右、上下翻转进行增强
    Returns:
    '''
    band_size = [0, ]
    images_concatenate = []
    for key in images:
        temp = np.transpose(images[key], (1, 2, 0))
        band_size.append(temp.shape[2] + band_size[-1])
        if key != 'seg':#根据传入的id进行修改


            aug = Compose([RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5) ], p=1)
        #     # ,
        #     # HueSaturationValue(hue_shift_limit=10,
            #                    sat_shift_limit=15,
            #                    val_shift_limit=10,
            #                    p=0.8), RGBShift(r_shift_limit=10,
            #                                     g_shift_limit=15,
            #                                     b_shift_limit=10,
            #                                     p=0.8),
            temp = temp.astype(np.uint8)
            temp = aug(image=temp)['image'].astype(np.float32)
        images_concatenate.append(temp)
    images_concatenate = np.concatenate(images_concatenate, axis=2)

    compose = Compose([RandomRotate90(p=0.5), Flip(p=0.5)], p=1)
    oneof = OneOf(
        [ShiftScaleRotate(shift_limit=(-0.2,0.2), scale_limit=(0.42,1.0), rotate_limit=0,
                                                              interpolation=cv2.INTER_LINEAR,
                                                              border_mode=cv2.BORDER_CONSTANT, p=0.8), ShiftScaleRotate(shift_limit=(-0.2,0.2), scale_limit=(0.42,1.0), rotate_limit=0,
                                                              interpolation=cv2.INTER_LINEAR,
                                                              border_mode=cv2.BORDER_CONSTANT, p=0.8)], p=1)
    images_concatenate = compose(image=images_concatenate)["image"]
    images_concatenate = oneof(image=images_concatenate)["image"]

    for i, key in enumerate(images):
        temp = images_concatenate[:, :, band_size[i]: band_size[i+1]]
        temp = np.transpose(temp, (2, 0, 1))
        images[key] = temp
    return images
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
            imagesA_dir=os.path.join(PATH_OUTPUT, 'A'),
            imagesB_dir=os.path.join(PATH_OUTPUT, 'B'),
            masks_dir=os.path.join(PATH_OUTPUT, "label"),
            augmentation=dataset.util.augmentation,
            preprocessing=preprocessing,
    ):
        self.imagesA_names = os.listdir(imagesA_dir)
        self.imageBs_names = self.imagesA_names
        self.masks_names = self.imagesA_names

        self.imagesA_fps = [os.path.join(imagesA_dir, name) for name in self.imagesA_names]
        self.imagesB_fps = [os.path.join(imagesB_dir, name) for name in self.imageBs_names]
        self.masks_fps = [os.path.join(masks_dir, name[:-4]+'.png') for name in self.masks_names]

        # convert str names to class values on masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        imageA = UNIT.img2numpy(self.imagesA_fps[i]).astype(np.float32)
        imageB = UNIT.img2numpy(self.imagesB_fps[i]).astype(np.float32)
        mask = UNIT.img2numpy(self.masks_fps[i]).astype(np.uint8)
        mask = np.where(mask == 0, 0, 1).astype(np.uint8)
        mask = np.expand_dims(mask, axis=0)

        # visualize(image=image, mask=mask)
        # apply augmentations
        if self.training:
            data = self.augmentation(Aimage=imageA,
                                     Bimage=imageB,
                                     seg=mask)
            imageA, imageB, mask = data["Aimage"], data["Bimage"], data["seg"]
        # apply preprocessing
        if self.preprocessing:
            image, mask = self.preprocessing(imageA=imageA,imageB=imageB, mask=mask)
        return imageA, imageB, mask

    def __len__(self):
        return len(self.imagesA_fps)


if __name__ == "__main__":
    '''
    运行整个程序前，先运行prepare，对Anhui数据进行分块处理。
    修改PATH_IMAGE_ROOT、PATH_IMAGES_SPEC、PATH_LABEL_IMAGES和PATH_OUTPUT以匹配用户自己的工作路径
    '''
    pass
