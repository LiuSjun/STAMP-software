##########################################
#  高分影像镶嵌

# filepathlist : 要镶嵌的影像路径
# savepath： 保存的路径

#####################################

import os
import math
import osgeo.gdal as gdal
import glob
from dataprocess.util  import *

def GetExtent(in_fn):
    '''
    计算影像角点的地理坐标或投影坐标
    ——————————————————————————
    @param：
        影像路径
    @return:
        min_x： x方向最小值
        max_y： y方向最大值
        max_x： x方向最大值
        min_y:  y方向最小值
    '''
    ds = gdal.Open(in_fn)
    geotrans = list(ds.GetGeoTransform())
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x = geotrans[0]
    max_y = geotrans[3]
    max_x = geotrans[0] + xsize * geotrans[1]
    min_y = geotrans[3] + ysize * geotrans[5]
    ds = None

    return min_x, max_y, max_x, min_y

def Mosaic_all (filepathlist, savepath):
    '''
    将指定路径文件夹下的tif影像全部镶嵌到一张影像上
    ————————————————————————————————
    '''
    filenum = len(filepathlist)
    in_files = filepathlist
    in_fn=in_files[0]
    #获取待镶嵌栅格的最大最小的坐标值
    min_x,max_y,max_x,min_y=GetExtent(in_fn)
    for in_fn in in_files[1:]:
        minx,maxy,maxx,miny=GetExtent(in_fn)
        min_x=min(min_x,minx)
        min_y=min(min_y,miny)
        max_x=max(max_x,maxx)
        max_y=max(max_y,maxy)

    #计算镶嵌后影像的行列号
    in_ds=gdal.Open(in_files[0])
    geotrans=list(in_ds.GetGeoTransform())
    width=geotrans[1]
    height=geotrans[5]

    columns=math.ceil((max_x-min_x)/width)
    rows=math.ceil((max_y-min_y)/(-height))
    in_band=in_ds.GetRasterBand(1)


    driver=gdal.GetDriverByName('GTiff')
    out_ds=driver.Create(savepath + 'raw_mosaiced_image.tif',columns,rows,3,in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0]=min_x
    geotrans[3]=max_y
    out_ds.SetGeoTransform(geotrans)

    #定义仿射逆变换
    inv_geotrans=gdal.InvGeoTransform(geotrans)

    for i in range(3):
        img_singleband = np.zeros((filenum,  rows,columns ), dtype=np.float32)
        out_band = out_ds.GetRasterBand(i+1)
        #开始逐渐写入
        for j in range (filenum):
            print('波段:'+ str(i+1) + ', 地区：' + str(j+1))
            in_ds=gdal.Open(in_files[j])

            in_gt=in_ds.GetGeoTransform()
            #仿射逆变换
            offset=gdal.ApplyGeoTransform(inv_geotrans,in_gt[0],in_gt[3])
            x,y=map(int,offset)
            #print(x,y)

            trans=gdal.Transformer(in_ds,out_ds,[])       #in_ds是源栅格，out_ds是目标栅格
            success,xyz=trans.TransformPoint(False,0,0)     #计算in_ds中左上角像元对应out_ds中的行列号
            x,y,z=map(int,xyz)
            #print(x,y,z)

            data=in_ds.GetRasterBand(i+1).ReadAsArray()
            height_, width_ = data.shape
            img_singleband[j, y:y+height_, x:x+width_ ] = data

        img_singleband = np.max(img_singleband, axis=0)
        out_band.WriteArray(img_singleband)



savepath = 'F:/西藏数据/日喀则市谢通门县/'
filepathlist = ['F:/西藏数据/日喀则市谢通门县/谢通门县01/Level16/谢通门县01.tif',
                'F:/西藏数据/日喀则市谢通门县/谢通门县02/Level16/谢通门县02.tif',
                'F:/西藏数据/日喀则市谢通门县/谢通门县03/Level16/谢通门县03.tif',
                'F:/西藏数据/日喀则市谢通门县/谢通门县04/Level16/谢通门县04.tif']

Mosaic_all(filepathlist,savepath)



