import os
import numpy as np
from osgeo import gdal
import time
import argparse
import UNIT

# 定义读取和保存图像的类
class GRID:

    def load_image(self, filename):
        image = gdal.Open(filename)

        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height)

        del image

        return img_proj, img_geotrans, img_data

    def write_image(self, filename, img_proj, img_geotrans, img_data):
        # 判断栅格数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判断数组维度
        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        # 创建文件
        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image  # 删除变量,保留数据

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='load remote sensing image and split to patch')
    parser.add_argument('--image_path',
                        default=r'E:\Newsegdataset\崇州\test\崇州市.tif',
                        help='remote sensing image path')
    parser.add_argument('--patch_size',
                        default=4,
                        help='patch size')
    parser.add_argument('--patch_save',
                        default=r'E:\Newsegdataset\崇州\test',
                        help='save path of patch image')
    args = parser.parse_args()
    print('待处理图像路径为:{}'.format(args.image_path))
    print('分块大小为:{}'.format(args.patch_size))
    print('分块图像保存路径:{}'.format(args.patch_save))
    image_path = args.image_path
    t_start = time.time()
    time_start = time.time()
    img_name = image_path
    proj, geotrans, data = GRID().load_image(img_name)

    # 图像分块
    patch_size = args.patch_size
    patch_save = args.patch_save
    channel, width, height = data.shape
    i_range = np.linspace(0, width, patch_size//2 + 1, endpoint=True, dtype=int)
    j_range = np.linspace(0, height, patch_size//2 + 1, endpoint=True, dtype=int)

    num = 0
    for i in range(len(i_range) - 1):
        for j in range(len(j_range) - 1):
            num += 1
            sub_image = data[:, i_range[i]:i_range[i + 1], j_range[j]:j_range[j + 1]]
            local_geotrans = list(geotrans)
            local_geotrans[0] = geotrans[0] + int(j_range[j]) * geotrans[1]
            local_geotrans[3] = geotrans[3] + int(i_range[i]) * geotrans[5]
            local_geotrans = tuple(local_geotrans)
            UNIT.numpy2img(os.path.join(patch_save, '{}.tif'.format(num)), sub_image, proj=proj, geot=local_geotrans)

            # GRID().write_image(os.path.join(patch_save, '{}.tif'.format(num)), proj, local_geotrans, sub_image)

    time_end = time.time()
    print('图像分块完毕, 耗时:{}秒'.format(round((time_end - time_start), 4)))
