from torch.utils.data import Dataset
import numpy as np
import os, UNIT, dataset.util
from tqdm import tqdm
import cv2
from osgeo import gdal

PATH_IMAGE_ROOT = r"D:\CropSegmentation\data\GF2\AH\Proccesed"
PATH_IMAGES_SPEC = [os.path.join(PATH_IMAGE_ROOT, "Subset{}.tif".format(i)) for i in range(3)]
PATH_IMAGES_LABEL = [os.path.join(PATH_IMAGE_ROOT, "Subset{}_object.dat".format(i)) for i in range(3)]
PATH_OUTPUT = r"D:\CropSegmentation\data\GF2\AH\Training"
SIZE, INTERVAL, RAND = 512, 400, 200


class SubsetGenerator:
    def __init__(self, path_img_spec, path_img_label, size, interval, rand):
        self.dts_spec = gdal.Open(path_img_spec)
        self.dts_label = gdal.Open(path_img_label)

        self.size = size
        self.interval = interval
        self.xlen, self.ylen = self.dts_spec.RasterXSize, self.dts_spec.RasterYSize
        self.x0_arr, self.y0_arr = self.get_xyoff()
        self.subset_img_spec = np.zeros((4, size, size))
        self.subset_img_label = np.zeros((size, size))
        self.rand = rand

    def __len__(self):
        return len(self.x0_arr) * len(self.y0_arr)

    def get_xyoff(self):
        x0_arr = np.arange(0, self.xlen - SIZE, self.interval, dtype=int)
        y0_arr = np.arange(0, self.ylen - SIZE, self.interval, dtype=int)
        return x0_arr, y0_arr

    def get_subset_img(self, idx):
        i, j = int(idx / len(self.x0_arr)), idx // len(self.x0_arr)

        x_rand, y_rand = np.random.randint(0, self.rand, 2)
        x0, y0 = self.x0_arr[i] + x_rand, self.y0_arr[j] + y_rand

        size_x_read, size_y_read = np.min((self.xlen - x0, self.size)), np.min((self.ylen - y0, self.size))

        size_x_read, size_y_read = int(size_x_read), int(size_y_read)
        x0, y0 = int(x0), int(y0)

        img_spec_readed = self.dts_spec.ReadAsArray(xoff=x0, yoff=y0, xsize=size_x_read, ysize=size_y_read)
        img_label_readed = self.dts_label.ReadAsArray(xoff=x0, yoff=y0, xsize=size_x_read, ysize=size_y_read)

        self.subset_img_spec[:], self.subset_img_label[:] = 0, 0

        self.subset_img_spec[:, :size_y_read, :size_x_read] = img_spec_readed
        self.subset_img_label[:size_y_read, :size_x_read] = img_label_readed

        return self.subset_img_spec, self.subset_img_label


def preprocessing(image, mask):
    image = image / 255
    return image, mask


class DatasetAH(Dataset):
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

    def __init__(
            self,
            path_images_spec=None,
            path_images_label=None,
            augmentation=dataset.util.augmentation,
            preprocessing=preprocessing,
    ):
        if path_images_label is None:
            path_images_label = PATH_IMAGES_LABEL
        if path_images_spec is None:
            path_images_spec = PATH_IMAGES_SPEC
        self.Generators = [SubsetGenerator(path_images_spec[i], path_images_label[i], SIZE, INTERVAL, RAND)
                           for i in range(len(path_images_spec))]
        self.len_arr = [self.Generators[i].__len__() for i in range(len(path_images_spec))]

        # convert str names to class values on masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        idx = 0
        for idx in range(len(self.len_arr)):
            if self.len_arr[idx] > i:
                break
            i -= self.len_arr[idx]
        img, mask = self.Generators[idx].get_subset_img(i)
        img = img / 1000

        mask = np.where(mask == 0, 0, 1).astype(np.uint8)
        mask = np.expand_dims(mask, axis=0)
        return img, mask

    def __len__(self):
        return np.sum(self.len_arr)


if __name__ == "__main__":
    '''
    运行整个程序前，先运行prepare，对Anhui数据进行分块处理。
    修改PATH_IMAGE_ROOT、PATH_IMAGES_SPEC、PATH_LABEL_IMAGES和PATH_OUTPUT以匹配用户自己的工作路径
    '''
    os.chdir(r"D:\CropSegmentation\data\GF2\AH\Proccesed")
    path_img_spec = "Subset0.tif"
    path_img_label = "Subset0_object.dat"

    sg = SubsetGenerator(path_img_spec, path_img_label, SIZE, INTERVAL, RAND)

    for i in range(10):
        image, mask = sg.get_subset_img(10)
        dataset.util.visual2(image, mask)
