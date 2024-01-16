from scipy.optimize import minimize
import numpy as np
import UNIT, cv2
import matplotlib.pyplot as plt
from numpy import random
from tqdm import tqdm
import datetime
from osgeo import gdal, ogr, osr
import os
import multiprocessing

def remove_noise(img, post_area_threshold=35, post_length_threshold=35):  #默认35
    contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   ##cv2.fineContours()先寻找外轮廓，返回每个轮廓contours[i]，个数及属性
    cc = np.zeros(img.shape)
    for i in tqdm(range(len(contours))):
        area = cv2.contourArea(contours[i])       ##计算轮廓面积
        length = cv2.arcLength(contours[i], True)    ##计算轮廓周长
        if area <= post_area_threshold or length <= post_length_threshold:
            cv2.drawContours(img, [contours[i]], 0, 0, -1)      ##cv2.drawContours()函数的功能是绘制轮廓
        else:
            cnt = contours[i]
            left = np.min((cnt[:, :, 1]))
            right = np.max((cnt[:, :, 1]))
            down = np.min((cnt[:, :, 0]))
            up = np.max((cnt[:, :, 0]))
            wit = right - left
            high = up - down
            rect = np.zeros((wit + 200, high + 200))
            cnt[:, :, 1] = cnt[:, :, 1] - left + 100
            cnt[:, :, 0] = cnt[:, :, 0] - down + 100
            cv2.fillPoly(rect, [cnt], 1)   ##填充任意形状的图型，可以用来绘制多边形。也可以使用非常多个边来近似的画一条曲线。cv2.fillPoly()函数可以一次填充多个图型。
            kernel_skeleton = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))  ##返回指定形状和尺寸的结构元素 MORPH_RECT矩形，表示方框的大小
            rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel_skeleton, iterations=1)  ##指的是先进行膨胀操作，再进行腐蚀操作
            cc[left:right, down: up] += rect[100:-100, 100:-100]

    return cc

def remove_noise_mutilpro(img):
    contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_list = []
    dd = np.zeros(img.shape)
    for i in range(0, len(contours), len(contours)//3):
        contours_list.append(contours[i:i + len(contours)//3])
    manager = multiprocessing.Manager()
    q = manager.Queue()
    process_name = []
    for i in range(len(contours_list)):
        process_name.append(
            multiprocessing.Process(target=pro_remove_noise,
                                    args=(img,contours_list[i], q)))
        process_name[i].start()

    for i in process_name:
        i.join()
    for j in process_name:
        aa = q.get()  # timeout=10
        dd[aa==1] = 1
    return dd
def pro_remove_noise(img,contours, q,post_area_threshold=35, post_length_threshold=35):
    cc = np.zeros(img.shape)
    for i in tqdm(range(len(contours))):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area <= post_area_threshold or length <= post_length_threshold:
            cv2.drawContours(img, [contours[i]], 0, 0, -1)
        else:
            cnt = contours[i]
            left =  np.min((cnt[:,:,1]))
            right = np.max((cnt[:,:,1]))
            down = np.min((cnt[:,:,0]))
            up = np.max((cnt[:, :, 0]))
            wit = right - left
            high = up - down
            rect = np.zeros((wit+200, high+200))
            cnt[:,:,1] = cnt[:,:,1]-left + 100
            cnt[:, :, 0] = cnt[:, :, 0] - down + 100
            cv2.fillPoly(rect, [cnt], 1)
            kernel_skeleton = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            rect = cv2.morphologyEx(rect, cv2.MORPH_CLOSE, kernel_skeleton, iterations=1)
            cc[left:right,down: up] += rect[100:-100,100:-100]
    q.put((cc), block=False)

def FillHole(im_in):
    # 复制 im_in 图像
    im_in = im_in.astype(np.uint8)
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv
    return im_out
def matthewsCorrelationCoefficient(pre, tar, threshold):
    ellipsis = 1e-1

    pre = (pre > threshold)
    tp = (tar * pre).sum().astype(np.double)
    tn = ((1 - tar) * (1 - pre)).sum().astype(np.double)
    fp = ((1 - tar) * pre).sum().astype(np.double)
    fn = (tar * (1 - pre)).sum().astype(np.double)

    N = tn + tp + fn + fp
    S = (tp + fn) / N
    P = (tp + fp) / N

    MCC = (tp/N - S*P) / np.sqrt(P * S * (1-S) * (1-P))
    return MCC


def seg_threshold():
    img_seg = UNIT.img2numpy("img_seg.tif")

    img_seg_real = UNIT.img2numpy("F:\project\Match\MatchMaterials\data\Preprocess\Subset1_object.dat")
    img_seg_real = np.where(img_seg_real == 0, 0, 1)

    optimal_fn = lambda t: 2 - matthewsCorrelationCoefficient(img_seg, img_seg_real, t)
    res = minimize(optimal_fn, np.array([0.4, ]), method="L-BFGS-B", bounds=((0.05, 0.95), ))
    print(res)

    arr = []
    ts = np.arange(0.4, 0.6, 0.01)
    for t in ts:
        arr.append(matthewsCorrelationCoefficient(img_seg, img_seg_real, t))

    plt.figure()
    plt.plot(ts, arr)
    plt.show()


def basic():
    img_bd = UNIT.img2numpy("img_resunet_bd.tif")
    img_dist = UNIT.img2numpy("img_resunet_dist.tif")
    img_seg = UNIT.img2numpy("img_resunet_seg.tif")

    thres_bd = 0.5
    thres_seg = 0.45
    thre_dist = 0.11

    img_bd_binary = (img_bd > thres_bd).astype(np.uint8)
    img_seg_binary = (img_seg > thres_seg).astype(np.uint8)
    background = ((img_seg_binary + img_bd_binary) < 1)
    img_seg_binary = img_seg_binary - (img_seg_binary * img_bd_binary)

    # img_dist = img_dist * (1 - img_bd) * img_seg_binary

    img_dist_binary = (img_dist > thre_dist).astype(np.uint8)
    ret, seed = cv2.connectedComponents(img_dist_binary)
    seed += 1
    seed[np.where(np.logical_and(seed == 1, background != 1))] = 0  # Setting Unknown Area
    seed = seed.astype(np.int32)
    UNIT.numpy2img("seed.tif", seed)

    # img_seg = img_seg_binary
    grey = ((img_seg - np.min(img_seg)) / (np.max(img_seg) - np.min(img_seg)) * 255).astype(np.uint8)
    rgb = np.zeros((grey.shape[0], grey.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = grey
    # rgb[np.where(rgb > 0)] = 1
    markers = cv2.watershed(rgb, seed)

    patch_loc = np.where(markers == 1)
    rgb[:, :, :] = 0
    rgb[patch_loc[0], patch_loc[1], :] = (0, 0, 0)
    cmp = plt.get_cmap("Paired")
    for i in tqdm(range(2, np.max(markers))):
        patch_loc = np.where(markers == i)
        color = np.array(cmp(np.random.randint(0, cmp.N)))
        color = color[:3] * 255
        rgb[patch_loc[0], patch_loc[1], :] = color
    rgb = np.transpose(rgb, (2, 0, 1))
    UNIT.numpy2img("B.tif", rgb)


def cropPatchLoss(img_ref, img_pre, t_size=50):
    np.random.seed(0)

    img_pre -= 1
    max_ref, max_pre = np.max(img_ref), np.max(img_pre)
    A = np.arange(1, max_ref, 1)
    marker_ref = random.choice(A, t_size)
    marker_pre = np.zeros(marker_ref.shape)
    value = np.zeros(t_size)

    for i in range(t_size):
        locs_ref = np.where(img_ref == marker_ref[i])
        # extend[:, i] = np.min(locs_ref[0]), np.max(locs_ref[0]), np.min(locs_ref[1]), np.max(locs_ref[1])
        count = np.bincount(img_pre[locs_ref])
        if len(count) == 1:  # 目标区域未被检测到
            continue
        count = count[1:]
        marker_pre[i] = np.argmax(count) + 1
        cross = np.where(np.logical_or(img_pre == marker_pre[i], img_ref == marker_ref[i]))[0].size
        overlap = np.where(np.logical_and(img_pre == marker_pre[i], img_ref == marker_ref[i]))[0].size
        value[i] = overlap / (cross + 1e-7)
    # print(value)
    return np.mean(value)


def cropPatch(img_dist, img_bd, img_seg, thre_dist, thres_bd, thres_seg):
    img_dist_binary = np.where(img_dist > thre_dist, 1, 0).astype(np.uint8)
    img_bd_binary = np.where(img_bd > thres_bd, 1, 0).astype(np.uint8)
    img_seg_binary = np.where(img_seg > thres_seg, 1, 0).astype(np.uint8)

    background = np.logical_and(np.logical_not(img_seg_binary), np.logical_not(img_bd_binary))
    # background = np.logical_not(img_seg_binary)
    _, seed = cv2.connectedComponents(img_dist_binary)

    img_seed = seed
    img_seed += 1  # Setting Background and Seed Location
    img_seed[np.where(np.logical_and(img_seed == 1, background != 1))] = 0  # Setting Unknown Area

    img_seg = ((img_seg - np.min(img_seg)) / (np.max(img_seg) - np.min(img_seg)) * 255).astype(np.uint8)
    img_seg = cv2.cvtColor(img_seg, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(img_seg, img_seed)
    markers[np.where(markers < 1)] = 1
    return markers

def RasettoShape(seg_path):
    start_time = datetime.datetime.now()
    inraster = gdal.Open(seg_path)  # 读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

    outshp = seg_path[:-4] + ".shp"  # 给后面生成的矢量准备一个输出文件名，这里就是把原栅格的文件名后缀名改成shp了
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    Poly_layer = Polygon.CreateLayer(seg_path[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
    newField = ogr.FieldDefn('value', ogr.OFTReal)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value,浮点型，
    Poly_layer.CreateField(newField)

    gdal.Polygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
    # gdal.FPolygonize(inband, None, Poly_layer, 0)  # 只转矩形，不合并
    Polygon.SyncToDisk()
    Polygon = None
    end_time = datetime.datetime.now()
    print("Succeeded at", end_time)
    print("Elapsed Time:", end_time - start_time)  # 输出程序运行所需时间

def mark2rgb(markers):
    xlen, ylen = markers.shape
    rgb = np.zeros((3, xlen, ylen), dtype=np.uint8)
    cmp = plt.get_cmap("Paired")

    color_index = np.zeros((np.max(markers) + 1, 3), dtype=np.uint8)
    for i in tqdm(range(np.max(markers))):
        color_index[i, :] = np.array(cmp(np.random.randint(0, cmp.N)))[:3] * 255

    for x in tqdm(range(xlen)):
        for y in range(ylen):
            if markers[x, y] <= 1:
                continue
            rgb[:, x, y] = color_index[markers[x, y]]
    return rgb

# from skimage import morphology
from torch.utils.data import random_split
if __name__ == "__main__":
    ROOT_PTN = r"E:\Newsegdataset\xizang\pths"
    ROOT_RST = r"E:\Newsegdataset\xizang\result"
    img_segnam= os.path.join(ROOT_RST, "3_seg.tif")
    img_bounam = os.path.join(ROOT_RST, "3_bou.tif")
    img_seg, proj, geot = UNIT.img2numpy(img_segnam, geoinfo=True)
    img_bou = UNIT.img2numpy(img_bounam)
    bound = 0.2   #0.2
    img_bou[img_bou <= bound] = 0
    img_bou[img_bou > bound] = 1

    thres_seg = 0.2   #0.3
    img_seg[img_seg >= thres_seg] = 1
    img_seg[img_seg < thres_seg] = 0
    img_seg[img_bou == 1] = 0


    # img_seg = remove_noise(img_seg.astype(np.uint8))
    # _, img_seg = cv2.connectedComponents(img_seg.astype(np.uint8))  ##对原始图中的每一个像素都打上标签，背景为0，连通域打上1，2，3。。。的标签，同一个连通域的像素打上同样的标签。相当与对每一个像素进行了分类（分割）
    rgb = mark2rgb(img_seg)
    UNIT.numpy2img(os.path.join(ROOT_RST, "3_result_gray.tif"), img_seg, proj=proj, geot=geot)  #

