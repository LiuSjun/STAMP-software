from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util
from tqdm import tqdm
import cv2


def preprocessing(image, mask):
    image = image / 255
    return image, mask


class DatasetBase(Dataset):
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
            root
    ):

        self.images_dir = os.path.join(root, "Spectral")
        self.masks_dir = os.path.join(root, "Label")
        self.dw_dir = os.path.join(root, "DW")

        self.images_names = os.listdir(self.images_dir)

        self.images_fps = [os.path.join(self.images_dir, name) for name in self.images_names]
        self.masks_fps = [os.path.join(self.masks_dir, name) for name in self.images_names]
        self.dw_fps = [os.path.join(self.dw_dir, name) for name in self.images_names]

    def __getitem__(self, i):
        # read data
        image = UNIT.img2numpy(self.images_fps[i]).astype(np.float32)
        dw = UNIT.img2numpy(self.dw_fps[i]).astype(np.float32)
        dw = np.expand_dims(dw, axis=0)

        mask = UNIT.img2numpy(self.masks_fps[i]).astype(np.uint8)
        mask = np.where(mask == 0, 0, 1).astype(np.uint8)
        mask = np.expand_dims(mask, axis=0)

        return image, mask, dw

    def __len__(self):
        return len(self.images_fps)


if __name__ == "__main__":
    '''
    运行整个程序前，先运行prepare，对Anhui数据进行分块处理。
    修改PATH_IMAGE_ROOT、PATH_IMAGES_SPEC、PATH_LABEL_IMAGES和PATH_OUTPUT以匹配用户自己的工作路径
    '''
    pass
