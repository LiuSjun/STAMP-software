from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util

PATH_OUTPUT = r"G:\FinalsForData\GF2\MS\Training"


def preprocessing(image, mask):
    image = image / 255
    return image, mask


class DatasetMS(Dataset):
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
