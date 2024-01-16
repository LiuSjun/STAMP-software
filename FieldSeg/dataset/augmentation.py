'''
该数据用于执行Image Augmentation
'''
import numpy as np
import random


def flip(image, mask, p=0.5):
    if random.random() > p:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    if random.random() > p:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=2)

    if random.random() > p:
        image = np.transpose(image, axes=(0, 2, 1))
        mask = np.transpose(mask, axes=(0, 2, 1))
    return image.copy(), mask.copy()


def colour_cast(image, mask, ap=0.2):
    bd, _, _ = image.shape
    for i_bd in range(bd):
        ap_bd = 2 * ap * random.random() - ap  # colour_cast changed from -ap to ap
        image[i_bd] += image[i_bd] * ap_bd
    return image, mask


class Operation:
    def __init__(self, p):
        pass
