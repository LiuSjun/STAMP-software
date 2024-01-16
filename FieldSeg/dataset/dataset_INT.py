'''
调用所有的Dataset,并构成一个新的Dataset
'''

from torch.utils.data import Dataset
from torch.utils.data import Subset
from dataset.datasets.dataset_ah import DatasetAH
from dataset.datasets.dataset_cd import DatasetCD
from dataset.datasets.dataset_gs import DatasetGS
from dataset.datasets.dataset_hlj import DatasetHLJ
from dataset.datasets.dataset_ms import DatasetMS
from dataset.datasets.dataset_zj import DatasetZJ
from dataset.datasets.dataset_zz import DatasetZZ
import os, numpy as np
import matplotlib.pyplot as plt
from dataset.augmentation import flip, colour_cast


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
        img, msk = dataset.arr_dts[i][locs[i]]
        visual(img, msk)


def visual(img, mask):
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, (2, 1, 0)]
    mask = mask[0, :, :]

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    datasets = (DatasetAH(), DatasetCD(), DatasetGS(), DatasetHLJ(), DatasetMS(), DatasetZJ(), DatasetZZ())
    dataset = DatasetGF2(
        datasets
    )

    # dataset_int_display(dataset)
    #
    # dataset_tra, dataset_val = dataset.dataset_split(0.9)
    # dataset_int_display(dataset_tra)
    # dataset_int_display(dataset_val)

    for i in range(10):
        image, mask = dataset[5]
        image, mask = flip(image, mask)
        visual(image, mask)
        image, mask = colour_cast(image, mask)
        visual(image, mask)