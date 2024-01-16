'''
使用决赛区域的extend数据进行MODEL的Validation
先使用Inference, 对结果进行推演。
再RUN本程序，进行精度统计。
'''

import matplotlib.pyplot as plt
import numpy as np

import UNIT, torch
import model.util as util_mod

metrics_fs = [util_mod.metric_f1, util_mod.metric_ac, util_mod.metric_mse,
              util_mod.metric_bce, util_mod.metric_tani]
metrics_names = ['F1', 'AC', 'MSE', 'BCE', 'Tani']


def image_validation(image, mask, metrics=None, metrics_names=None):
    '''
    输入Inference和Label
    统计各项指标
    '''
    image = torch.from_numpy(image).to(torch.float32)
    mask = torch.from_numpy(mask).to(torch.float32)

    metrics_out = [metric(image, mask).cpu() for metric in metrics]
    for i in range(len(metrics)):
        print(metrics_names[i], ":", metrics_out[i], end='; ')
    return


if __name__ == "__main__":
    path_mask = r"Z:\Crop-Competition\决赛数据\TrainingData\区域2\区域2_Label.dat"

    mask = UNIT.img2numpy(path_mask)

    print("Nir RGB")
    path_image = r"D:\CropSegmentation\result\SUB区域2_RST_LSJ_training.tif"
    image = UNIT.img2numpy(path_image)
    image_validation(image, mask, metrics_fs, metrics_names)
    print()

    # print("RGB")
    # path_image = r"D:\CropSegmentation\result\SUB区域2_RST_RGB.tif"
    # image = UNIT.img2numpy(path_image)
    # image_validation(image, mask, metrics_fs, metrics_names)

