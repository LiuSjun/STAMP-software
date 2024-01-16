'''
该代码用于学习mmsegmentation框架
'''

import os
from tools import train

if __name__ == "__main__":
    path = "D:\CropSegmentation\data\mmseg_data\PSPNet_config.py"
    work_dir = r"D:\CropSegmentation\data\mmseg_data"

    train.main()
    pass
