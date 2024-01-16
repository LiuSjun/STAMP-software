from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util
from tqdm import tqdm
import cv2

PATH_OUTPUT = r"G:\ChangeDete\test_AB"


def preprocessing(image, mask):
    image = image / 255
    return image, mask


class DatasetChange(Dataset):
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
            imagesA_dir=os.path.join(PATH_OUTPUT, "A"),
            imagesB_dir=os.path.join(PATH_OUTPUT, "B"),
            masks_dir=os.path.join(PATH_OUTPUT, "scoreHR4"),
            masksr_dir=os.path.join(PATH_OUTPUT, "resultHR4"),
            augmentation=dataset.util.augmentation,
            preprocessing=preprocessing,
    ):
        self.images_names = os.listdir(imagesA_dir)
        self.masks_names = self.images_names

        self.imagesA_fps = [os.path.join(imagesA_dir, name) for name in self.images_names]
        self.imagesB_fps = [os.path.join(imagesB_dir, name) for name in self.images_names]
        self.masks_fps = [os.path.join(masks_dir, name[:-4]+'.png') for name in self.masks_names]
        self.masksr_fps = [os.path.join(masksr_dir, name[:-4] + '.png') for name in self.masks_names]

        # convert str names to class values on masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        imageA = UNIT.img2numpy(self.imagesA_fps[i]).astype(np.float32)
        imageB = UNIT.img2numpy(self.imagesB_fps[i]).astype(np.float32)
        mask = self.masks_fps[i]
        maskr = self.masksr_fps[i]
        image = np.concatenate((imageA,imageB),axis=0)
        # visualize(image=image, mask=mask)
        # apply augmentations
        # apply preprocessing
        if self.preprocessing:
            image, mask = self.preprocessing(image=image, mask=mask)
        return image, mask, maskr

    def __len__(self):
        return len(self.imagesA_fps)


if __name__ == "__main__":
    '''
    运行整个程序前，先运行prepare，对Anhui数据进行分块处理。
    修改PATH_IMAGE_ROOT、PATH_IMAGES_SPEC、PATH_LABEL_IMAGES和PATH_OUTPUT以匹配用户自己的工作路径
    '''
    pass
