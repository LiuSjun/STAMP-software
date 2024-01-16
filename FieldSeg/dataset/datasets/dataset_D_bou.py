from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util
from tqdm import tqdm
import cv2

PATH_IMAGE_ROOT = r"G:\FinalsForData\GF2\AH\Proccesed"
PATH_IMAGES_SPEC = [os.path.join(PATH_IMAGE_ROOT, "Subset{}.tif".format(i)) for i in range(3)]
PATH_IMAGES_LABEL = [os.path.join(PATH_IMAGE_ROOT, "Subset{}_object.dat".format(i)) for i in range(3)]
PATH_OUTPUT = r"E:\Newsegdataset\xizang\Dong\Training"


def preprocessing(image, mask):
    image = image / 255
    return image, mask


class DatasetD(Dataset):
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
            masks_dir=os.path.join(PATH_OUTPUT, "Boundary"),
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
    pass
