import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from osgeo import gdal, osr
import numpy as np
import os
from tqdm import tqdm, trange
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# 用来显示掩膜
def img2numpy(file_path, geoinfo=False):
    assert os.path.exists(file_path), "NO FILE!!! {}".format(file_path)
    dts = gdal.Open(file_path)
    proj = dts.GetProjection()
    geot = dts.GetGeoTransform()
    img = dts.ReadAsArray()
    if geoinfo:
        return img, proj, geot
    else:
        return img
def normalize_parameters(img):
    '''
    获取影像各个波段的normalize parameters
    '''
    img = np.where(img == 0, np.nan, img)
    top = np.nanpercentile(img, 99, axis=(0,1))
    bottom = np.nanpercentile(img, 1, axis=(0,1))
    return top, bottom
def normalize_apply(img, paras):
    para_top, para_bottom = paras
    for i in range(len(para_top)):
        img_bnd = img[:, :, i]
        top, bottom = para_top[i], para_bottom[i]

        img_bnd[img_bnd > top] = top
        img_bnd[img_bnd < bottom] = bottom
        img_bnd = (img_bnd - bottom) / (top - bottom) * 255
        img[:, :, i] = img_bnd
    img = img.astype("uint8")
    return img
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.65]])
        img[m] = color_mask
    ax.imshow(img)


sam_checkpoint = r"F:\pths\sam_vit_b_01ec64.pth" # 模型
model_type = "vit_b"
numstrid = 64
image = img2numpy(r'C:\Users\Administrator\Desktop\temp\SAM\Diff\PlanetsmallRgb.tif').transpose((1, 2, 0))[:,:,:3]
# 创建可视化图像并保存
save_path = r'C:\Users\Administrator\Desktop\temp\SAM\Diff\PlanetsmallRgb_SAM.tif' # 您的保存路径
# image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
image = normalize_apply(image, normalize_parameters(image))
# imagetest = img2numpy(r'F:\pths\data\GEE_S2.tif')
DW = np.ones((2000, 2000))
# 实验测试数据处理流程
# DW[DW<100] = 0
# DW[DW>=100] = 1

# 正常数据处理流程
# DW[DW!=4] = 1
# DW[DW==4] = 0

# 定义膨胀和腐蚀的结构元素
kernel = np.ones((20, 20), np.uint8)  # 3x3 方形结构元素

# 膨胀操作
dilated_image = cv2.dilate(DW, kernel, iterations=1)
# # 腐蚀操作
# eroded_DW = cv2.erode(dilated_image, kernel, iterations=1)
dilated_image = 1 - dilated_image


from scipy.ndimage import sobel
# 使用 sobel() 函数计算梯度图
gradient_x = sobel(image, axis=1)
gradient_y = sobel(image, axis=0)
gradient = np.sqrt(gradient_x**2 + gradient_y**2).astype(np.uint8)[:,:,0]
# gradient = cv2.normalize(gradient, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

device = "cuda" # 使用GPU

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


# 有限范围内的自动采样点的设计
# point_grids =[np.array([[126/912,596/911],[529/912,77/911],[111/912,795/911],[842/912,882/911]]) # 通过手动增加设置点来进行完善
# 设计函数，完成在有限范围内的自动采样,均匀采样
def define_area_average_Field(mask, n_per_side):
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    w, h = mask.shape

    # stride_w = w // (n_per_side + 1)
    # stride_h = h // (n_per_side + 1)
    # sampled_mask =  mask[int(1 / (2 * n_per_side) * w)::stride_w, int(int(1 / (2 * n_per_side) * h))::stride_h]
    # mask_points = sampled_mask[:n_per_side, :n_per_side].reshape(-1, 1)
    # filtered_matrix = points * mask_points
    pointX = points[:,0] * w
    pointY = points[:, 1] * h
    point_image = np.array([pointX,pointY])
    point_image = point_image.transpose((1,0)).astype(np.int16)
    mask_points = [mask[coord[0],coord[1]] for coord in point_image]
    mask_points = np.array(mask_points).reshape(-1, 1)
    filtered_matrix = points * mask_points
    filtered_matrix_zeros = filtered_matrix[np.logical_not(np.all(filtered_matrix == [0, 0], axis=1))]
    filtered_matrix_zeros = np.fliplr(filtered_matrix_zeros)
    pointX = filtered_matrix_zeros[:,0] * w
    pointY = filtered_matrix_zeros[:, 1] * h
    point_image = np.array([pointX, pointY]).transpose((1, 0))
    return point_image
def random_points_from_matrix(matrix, num_points):
    """
    Randomly selects num_points from the given matrix and returns their coordinates
    in a normalized n*2 matrix.
    """
    # 获取矩阵的维度
    rows, cols = matrix.shape

    # 生成所有可能的坐标
    all_coords = [(i, j) for i in range(rows) for j in range(cols)]

    # 随机选择 num_points 个坐标
    selected_coords = np.array(np.random.choice(len(all_coords), num_points, replace=False))

    # 转换为对应的行列坐标
    coords = np.array([all_coords[i] for i in selected_coords])

    # 归一化
    # normalized_coords = coords / np.array([rows - 1, cols - 1])

    return coords
crop_n_points_downscale_factor = 1
stride = [numstrid]
Total_Seed = []
point_number = 3000
for i in stride:
    print(i)
    Total_Seed.append(define_area_average_Field(DW, i))
    # Total_Seed.append(random_points_from_matrix(DW, point_number))


masks2 = SamPredictor(sam)
masks2.set_image(image)
masks_list = []
score_lsit = []
for point in tqdm(Total_Seed[0]):
    # point = [399,440] # 第一个测试点 [450, 400]; 第二个测试点 [450, 350]; 第三个测试点 [450, 450];

    input_point = np.array([point])
    input_label = np.array([1])
    masks1, scores1, logits = masks2.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
        )

    # 将得分和掩膜绑定在一起并按得分排序
    scored_masks = sorted(zip(scores1, masks1), key=lambda x: x[0], reverse=True)

    # 遍历排序后的掩膜列表，选择合适的掩膜
    selected_mask = None
    max_score = -1

    for score, mask in scored_masks:
        if np.sum(mask) < 25000:
            selected_mask = mask
            max_score = score
            break

    # 如果没有找到面积小于25000的掩膜，选择得分最高的掩膜
    if selected_mask is None and scored_masks:
        print("No suitable mask found")
        continue
    # 使用形态学闭操作去除孔洞
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(selected_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 连通组件分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, 8, cv2.CV_32S)

    # 检查是否存在除背景之外的其他连通区
    if num_labels > 1:
        # 选择面积最大的区域（忽略背景）
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = (labels == largest_label)
    else:
        # 没有找到有效的连通区，可以选择跳过当前掩膜，或者使用某种默认处理
        print("No valid connected components found.")
        continue  # 或者选择其他操作

    masks_list.append(largest_mask)
    score_lsit.append(max_score)

# 进行田块得分的排序后合并
mask_num = np.arange(len(score_lsit))
sorted_tuples = sorted(zip(score_lsit, mask_num))
score_lsit, mask_num = zip(*sorted_tuples)

# 以得分从高到低排序田块
sorted_indices = np.argsort(score_lsit)[::-1]

# 初始化总掩膜和彩色掩膜
All_mask = np.zeros_like(masks_list[0], dtype=bool)
Number_mask = np.zeros((All_mask.shape[0], All_mask.shape[1], 3), dtype=np.uint8)

# 遍历排序后的田块
# 田块编码
count_num = 1
for idx in sorted_indices:
    current_mask = masks_list[idx]
    current_score = score_lsit[idx]

    # 只考虑得分高于阈值的田块
    if current_score > 0.10:
        # 计算与已有田块的交集
        intersection = np.logical_and(All_mask, current_mask)
        intersection_ratio = np.sum(intersection) / np.sum(current_mask)

        # 如果交叉比小于10%，合并田块
        if intersection_ratio < 0.1:
            All_mask = np.logical_or(All_mask, current_mask)

            # 为当前地块分配一个随机颜色
            random_color = np.random.randint(0, 255, 3)
            Number_mask[current_mask] = random_color
            count_num += 1

# 将最终的掩膜转换为彩色以进行显示
# Number_mask[All_mask] = [255, 255, 255]  # 白色


atotal = Total_Seed[0]
atotalX = atotal[:,0]
atotalY = atotal[:,1]

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(image)
# plt.plot(atotalX, atotalY, 'ro')
plt.axis('off')

plt.subplot(122)
plt.imshow(image)
plt.imshow(Number_mask)
plt.axis('off')
plt.show()


cv2.imwrite(save_path, cv2.cvtColor(Number_mask, cv2.COLOR_RGB2BGR))

# threshold = 10000
# masks2 = [item for item in masks2 if item['area'] <= threshold]


