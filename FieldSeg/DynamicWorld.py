'''
决定了，可能的操作直接在ENVI上进行。
1. 直接把数据在DynamicWorld上裁剪下来，记为DW.tif (ENVI File Save as)
2. 把DW.tif的投影转换为MSS格式的投影 (ENVI reproject Raster)
3. 对两个数据集直接进行配准
4. 将DW.tif的分辨率也转换为与MSS一致
5. 在MSS上，裁剪出对应的DW.tif数据集
6. 当DW.tif涉及到多个区域时，需要进行区域合并
'''

import numpy as np, os
from osgeo import gdal, osr

import UNIT

ROOT_DynamicWorld = r"Z:\LSJData\DynamicWorld\download"
FILENAMES_DynamicWorld = ["DSS.dat", "DSS_other.dat", "ChongqingHubei",
                          "GZ.dat", "HZ.dat", "shanghai", "Xizang",
                          "ChongqingHubei", "NMG.dat", "NMG_other.dat",
                          "QGN.dat", "shanghai", "Shanxi", "Xizang",
                          "XJ.dat", "YN.dat", "sichuan"]

# 以下文件均需要进行配准
FILENAMES_AREA = [r"D:\CropSegmentation\data\GF2\AH\Proccesed\Subset0.tif",
                  r"D:\CropSegmentation\data\GF2\AH\Proccesed\Subset1.tif",
                  r"D:\CropSegmentation\data\GF2\AH\Proccesed\Subset2.tif",
                  r"D:\CropSegmentation\data\GF2\CD\Processed\ChenduSubRaster.tif",
                  r"D:\CropSegmentation\data\GF2\HLJ\Proccesed\GF2_20170731_Subset0.dat",
                  r"D:\CropSegmentation\data\GF2\HLJ\Proccesed\GF2_20170731_Subset1.dat",
                  r"D:\CropSegmentation\data\GF2\ZJ\Processed\Sub0.tif",
                  r"D:\CropSegmentation\data\GF2\ZJ\Processed\Sub1.tif",
                  r"D:\CropSegmentation\data\GF2\ZJ\Processed\Sub2.tif",
                  r"D:\CropSegmentation\data\GF2\ZJ\Processed\Sub3.tif",
                  r"D:\CropSegmentation\data\GF2\ZJ\Processed\Sub4.tif",
                  r"D:\CropSegmentation\data\GF2\MS\sub1ps.tif",
                  r"D:\CropSegmentation\data\GF2\MS\sub2ps.tif",
                  r"D:\CropSegmentation\data\GF2\GS\sub1_20200520.tif",
                  r"D:\CropSegmentation\data\GF2\GS\sub2_20200520.tif",
                  r"D:\CropSegmentation\data\GF2\ZZ\sub1.tif",
                  r"D:\CropSegmentation\data\GF2\ZZ\sub2.tif",
                  r"Z:\Crop-Competition\决赛数据\Crop_C\MAP杯数智农业大赛地块识别初赛辅助数据\区域1-MSS1.tif",
                  r"Z:\Crop-Competition\决赛数据\Crop_C\MAP杯数智农业大赛地块识别初赛辅助数据\区域2-MSS2.tif",
                  r"Z:\Crop-Competition\决赛数据\Crop_C\MAP杯数智农业大赛地块识别初赛辅助数据\区域3-MSS1.tif",
                  r"Z:\Crop-Competition\决赛数据\Crop_C\MAP杯数智农业大赛地块识别初赛辅助数据\区域4-MSS1.tif"]


os.chdir(ROOT_DynamicWorld)


def region_clip(dts, ext, output):
    '''
    :param dts: 要裁剪的数据集
    :param ext: 裁剪的范围
    :param output: 裁剪完成后，输出的文件
    :return:
    '''
    ext_dts = get_extend(dts)
    p0, l0 = UNIT.xy2pl((ext_dts[0], ext_dts[2]), dts.GetGeoTransform())
    p1, l2 = UNIT.xy2pl((ext_dts[1], ext_dts[3]), dts.GetGeoTransform())
    print(p0, l0)
    print(p1, l2)


def img_clip(arr_dts, ext=None):
    for i in range(len(FILENAMES_DynamicWorld)):
        ext_tp = get_extend(arr_dts[i])
        # 以下参数成立，意味着裁剪区域不在该dts中
        if ext_tp[0] > ext[1] or ext_tp[1] < ext[0] or ext_tp[2] < ext[3] or ext_tp[3] > ext[2]:
            continue
        # print("", ext_tp)
        # print("", ext)
        # region_clip(arr_dts[i], ext, None)
        # print("Find ", FILENAMES_DynamicWorld[i])
        return
    # print("Can not Find, Please Check")


def get_extend(dts):
    proj, geot = dts.GetProjection(), dts.GetGeoTransform()

    x0, xres, y0, yres = geot[0], geot[1], geot[3], geot[5]
    xlen, ylen = dts.RasterXSize, dts.RasterYSize
    x1, y1 = x0 + xlen * xres, y0 + ylen * yres

    if x0 > 1000:  # 若大于1000，则说明x1为投影坐标系
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(proj)
        geosrs = prosrs.CloneGeogCS()
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        coords0 = ct.TransformPoint(x0, y0)
        coords1 = ct.TransformPoint(x1, y1)
        x0, x1 = coords0[1], coords1[1]
        y0, y1 = coords0[0], coords1[0]
    return x0, x1, y0, y1


# 读取所有Image的经纬度范围，并保存其Dataset文件
def region_reader():
    arr_dts = []
    arr_ext = []
    for i in range(len(FILENAMES_DynamicWorld)):

        dts_tp = gdal.Open(FILENAMES_DynamicWorld[i])
        arr_dts.append(dts_tp)

        # proj, geot = dts_tp.GetProjection(), dts_tp.GetGeoTransform()
        # arr_pg.append((proj, geot, ))

        # arr_ext.append(get_extend(dts_tp))
    return arr_dts



if __name__ == "__main__":
    # arr_dts = region_reader()
    # print(arr_dts)
    #
    # for path_area in FILENAMES_AREA:
    #     ext = get_extend(gdal.Open(path_area))
    #     print(ext)
    #     img_clip(arr_dts, ext)

    # # AH
    # path_dw = r"D:\CropSegmentation\data\GF2\AH\dw\dw.dat"
    # path_training = r"D:\CropSegmentation\data\GF2\AH\Processed\Subset0.tif"
    # path_output = r"D:\CropSegmentation\data\GF2\AH\Processed\Subset0_DW.tif"
    # UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)
    # path_training = r"D:\CropSegmentation\data\GF2\AH\Processed\Subset1.tif"
    # path_output = r"D:\CropSegmentation\data\GF2\AH\Processed\Subset1_DW.tif"
    # UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)
    # path_training = r"D:\CropSegmentation\data\GF2\AH\Processed\Subset2.tif"
    # path_output = r"D:\CropSegmentation\data\GF2\AH\Processed\Subset2_DW.tif"
    # UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)

    # CD
    path_dw = r"D:\CropSegmentation\data\GF2\CD\dw\dw.dat"
    path_training = r"D:\CropSegmentation\data\GF2\CD\Processed\ChenduSubRaster.tif"
    path_output = r"D:\CropSegmentation\data\GF2\CD\Processed\ChenduSubRaster_DW.tif"
    UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)

    # GS
    path_dw = r"D:\CropSegmentation\data\GF2\GS\dw\dw_sub1.dat"
    path_training = r"D:\CropSegmentation\data\GF2\GS\Processed\sub1_20200520.tif"
    path_output = r"D:\CropSegmentation\data\GF2\GS\Processed\sub1_DW.tif"
    UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)
    path_dw = r"D:\CropSegmentation\data\GF2\GS\dw\dw_sub2.dat"
    path_training = r"D:\CropSegmentation\data\GF2\GS\Processed\sub2_20200520.tif"
    path_output = r"D:\CropSegmentation\data\GF2\GS\Processed\sub2_DW.tif"
    UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)

    # HLJ
    path_dw = r"D:\CropSegmentation\data\GF2\HLJ\dw\dw_warp.dat"
    path_training = r"D:\CropSegmentation\data\GF2\HLJ\Processed\GF2_20170731_Subset0.dat"
    path_output = r"D:\CropSegmentation\data\GF2\HLJ\Processed\GF2_20170731_Subset0_DW.tif"
    UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)
    path_training = r"D:\CropSegmentation\data\GF2\HLJ\Processed\GF2_20170731_Subset1.dat"
    path_output = r"D:\CropSegmentation\data\GF2\HLJ\Processed\GF2_20170731_Subset1_DW.tif"
    UNIT.raster_clip(path_dw, path_training, path_output, datatype='GTIFF', invalid=None)
