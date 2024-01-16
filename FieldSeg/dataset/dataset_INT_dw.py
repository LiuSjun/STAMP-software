'''
调用所有的Dataset,并构成一个新的Dataset
'''

from torch.utils.data import Dataset
from torch.utils.data import Subset
from .dataset_dw.dataset_base import DatasetBase
import os, numpy as np
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Anaconda 的包和Pytorch的包冲突，在Figure（）时出错


class DatasetGF2(Dataset):
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
            arr_dts
    ):
        self.arr_dts = arr_dts
        self.len = []
        for dts in arr_dts:
            self.len.append(len(dts))
        self.len = np.array(self.len)

    def __getitem__(self, i):
        idx = 0
        for idx in range(len(self.len)):
            if self.len[idx] > i:
                break
            i -= self.len[idx]
        return self.arr_dts[idx][i]

    def __len__(self):
        return np.sum(self.len)

    def num_dts(self):
        return len(self.arr_dts)

    def dataset_split(self, pre):
        '''
        pre 为训练集的比例，0 - 1
        '''
        lens_tra = (self.len * pre).astype(int)
        dataset_tra = [Subset(self.arr_dts[i], np.arange(0, lens_tra[i], 1)) for i in range(self.num_dts())]
        dataset_val = [Subset(self.arr_dts[i], np.arange(lens_tra[i], self.len[i], 1)) for i in range(self.num_dts())]
        return DatasetGF2(dataset_tra), DatasetGF2(dataset_val)


def dataset_int_display(dataset):
    num_datasets = dataset.num_dts()

    locs = np.random.randint(0, np.min(dataset.len), num_datasets)
    for i in range(num_datasets):
        img, msk, dw = dataset.arr_dts[i][locs[i]]
        visual(img, msk, dw)


def visual(img, mask, dw):
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, (2, 1, 0)] / 255

    mask = mask[0, :, :]
    dw = dw[0, :, :]

    plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(dw)
    plt.show()


if __name__ == "__main__":

    datasets = (DatasetBase(r"D:\CropSegmentation\data\GF2\AH\Training"),
                DatasetBase(r"D:\CropSegmentation\data\GF2\CD\Training"),
                DatasetBase(r"D:\CropSegmentation\data\GF2\GS\Training"),
                DatasetBase(r"D:\CropSegmentation\data\GF2\HLJ\Training"))
    dataset = DatasetGF2(
        datasets
    )
    dataset_int_display(dataset)

    dataset_tra, dataset_val = dataset.dataset_split(0.9)
    dataset_int_display(dataset_tra)
    dataset_int_display(dataset_val)