import tkinter as tk
from tkinter import ttk, filedialog, Scale
from PIL import Image, ImageTk
from PIL import Image, ImageTk
import numpy as np
from osgeo import gdal, osr
import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
from tqdm import tqdm
from ttkthemes import ThemedTk  # 引入更现代的主题
from osgeo import gdal


def resource_path(relative_path):
    """获取资源的绝对路径。用于PyInstaller打包后资源的访问。"""
    try:
        # PyInstaller创建的临时文件夹
        base_path = sys._MEIPASS
    except Exception:
        # 正常的Python环境
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def splash_screen():
    # 启动界面
    splash = tk.Tk()
    splash.overrideredirect(True)  # 隐藏窗口边框
    splash.geometry("800x600+300+200")  # 设置启动界面的大小和位置

    # 加载图片
    logo_path = resource_path("Logo.png")
    image = Image.open(logo_path)  # 替换为您的图片路径
    photo = ImageTk.PhotoImage(image)

    # 创建一个带有图片的标签
    splash_label = tk.Label(splash, image=photo)
    splash_label.pack(expand=True)

    # 启动界面显示时间（例如 3000 毫秒 = 3 秒）
    splash.after(3000, splash.destroy)

    splash.mainloop()
# 先显示启动界面
splash_screen()

# 全局变量 模式切换
mode = "click"  # 可以是 "click" 或 "drag"
# SAM 模型定义为全局变量
masks2 = None
image1 = None
dragging = False
# 全局变量来存储图像历史
image_history = []
# 全局变量来存储最后的图像对象
last_image = None
# 假设image1是一个形状为(height, width, channels)的NumPy数组
update_mask = None

# 全局变量来存储地理参考和投影信息
geo_transform = None
projection = None

# 存储不同图层的图像数据
layer_images = {}
selected_layer = None

# 存储图层透明度信息
layer_opacity = {}

# 全局变量来存储图层顺序
layer_order = []

# 手动勾画mask全局
polygon_points = []
mask_manual = None

# 在顶部定义全局变量
sam_checkpoints = ['Fast', 'Medium', 'Slow']
model_types = ["vit_b", 'vit_l', 'vit_h']
selected_sam_checkpoint = sam_checkpoints[0]  # 默认选择为 'Fast'
selected_model_type = model_types[0]  # 默认选择

# AutoSAM的超参
numstrid = 64  # 初始值，可以根据需要调整


def clear_all_data():
    global image1, masks2, last_image, update_mask, layer_images, layer_order, image_history, mask_manual
    # 清除影像数据和掩膜
    image1 = None
    masks2 = None
    update_mask = None
    mask_manual = None
    last_image = None
    layer_images.clear()
    layer_order.clear()
    image_history.clear()  # 清除历史记录

    # 清除画布
    canvas.delete("all")



# 更新 numstrid 的函数
def update_numstrid():
    global numstrid
    try:
        numstrid = int(numstrid_entry.get())  # 从输入框获取数字并转换为整数
    except ValueError:
        print("请输入有效的数字")

def update_model_selection(event):
    global selected_sam_checkpoint, selected_model_type
    index = model_combobox.current()
    selected_sam_checkpoint = sam_checkpoints[index]
    selected_model_type = model_types[index]

    # 根据选择的模型速度，更新 SAM 模型路径
    if selected_sam_checkpoint == 'Fast':
        selected_sam_checkpoint = 'sam_vit_b_01ec64.pth'
    elif selected_sam_checkpoint == 'Medium':
        selected_sam_checkpoint = 'sam_vit_l_0b3195.pth'
    else: # Slow
        selected_sam_checkpoint = 'sam_vit_h_4b8939.pth'

def update_opacity_scale():
    global selected_layer, layer_opacity
    if selected_layer in layer_opacity:
        opacity_scale.set(layer_opacity[selected_layer])
    else:
        opacity_scale.set(100)  # 默认透明度100%

def update_layer_opacity(value):
    global selected_layer, layer_opacity
    if selected_layer:
        layer_opacity[selected_layer] = int(value)
        update_canvas()

def open_image():
    global image1, photo, image_id, image, masks2, last_image, image_history, update_mask
    global geo_transform, projection
    global selected_sam_checkpoint, selected_model_type
    # 确保 selected_sam_checkpoint 是有效的文件名
    if selected_sam_checkpoint in ['Fast', 'Medium', 'Slow']:
        if selected_sam_checkpoint == 'Fast':
            selected_sam_checkpoint = 'sam_vit_b_01ec64.pth'
        elif selected_sam_checkpoint == 'Medium':
            selected_sam_checkpoint = 'sam_vit_l_0b3195.pth'
        else:  # Slow
            selected_sam_checkpoint = 'sam_vit_h_4b8939.pth'


    file_path = filedialog.askopenfilename(title="Select an Image for IAF extraction",
                                           filetypes=[("Image files", "*.jpg *.tif *.png *.bmp *.gif")])

    # 使用GDAL打开影像并获取地理参考和投影信息
    dataset = gdal.Open(file_path)
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    if file_path:
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=photo, anchor='nw')
        canvas.config(scrollregion=canvas.bbox("all"))

        # 创建SAM对象
        image1 = np.array(image)

        device = "cuda"  # 使用GPU
        sam = sam_model_registry[selected_model_type](checkpoint=selected_sam_checkpoint)
        sam.to(device=device)
        masks2 = SamPredictor(sam)
        masks2.set_image(image1)

        # 假设image1是一个形状为(height, width, channels)的NumPy数组
        update_mask = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]), dtype=np.uint8)

        # 添加图像到图层字典
        layer_images["Original Image"] = Image.fromarray(image1)
        layer_order.append("Original Image")  # 添加到图层顺序列表

        # 更新图层选择下拉列表
        update_layer_combobox()

def update_layer_combobox():
    layer_combobox['values'] = list(layer_images.keys())

def update_canvas():
    global selected_layer, layer_images, photo, last_image, image1
    # if selected_layer in layer_images:
    #     # 获取选定图层的图像
    #     image = layer_images[selected_layer].copy()
    #     opacity = layer_opacity.get(selected_layer, 100) / 100  # 默认透明度100%
    #     if opacity < 1:
    #         alpha = int(255 * opacity)
    #         image.putalpha(alpha)  # 设置透明度
    #     photo = ImageTk.PhotoImage(image)
    #     canvas.create_image(0, 0, image=photo, anchor='nw')
    #     canvas.config(scrollregion=canvas.bbox("all"))
    #     last_image = photo  # 保存对photo的引用

    if not layer_order:
        return

    # 创建一个透明的底图
    composite_image = Image.new("RGBA", (image1.shape[1], image1.shape[0]), (0, 0, 0, 0))

    for layer_name in layer_order:
        if layer_name in layer_images:
            layer_image = layer_images[layer_name].convert("RGBA")
            opacity = layer_opacity.get(layer_name, 100) / 100
            alpha = int(255 * opacity)
            layer_image.putalpha(alpha)
            composite_image = Image.alpha_composite(composite_image, layer_image)

    photo = ImageTk.PhotoImage(composite_image)
    canvas.create_image(0, 0, image=photo, anchor='nw')
    canvas.config(scrollregion=canvas.bbox("all"))
    last_image = photo

def on_select(event):
    global selected_layer
    selected_layer = layer_var.get()
    # 将当前选择的图层移至图层顺序列表的末尾（顶层）
    if selected_layer in layer_order:
        layer_order.remove(selected_layer)  # 先移除旧位置
    layer_order.append(selected_layer)  # 再添加到末尾

    # 重置选择的图层的透明度为 100%
    layer_opacity[selected_layer] = 100
    opacity_scale.set(100)  # 更新透明度滑块的值

    # update_opacity_scale()
    update_canvas()

# 计算偏移量
def get_scroll_offset():
    x_view = scroll_x.get()
    y_view = scroll_y.get()

    scrollregion = canvas.cget("scrollregion").split()
    scrollregion = list(map(int, scrollregion))

    x_offset = int((scrollregion[2] - scrollregion[0]) * x_view[0])
    y_offset = int((scrollregion[3] - scrollregion[1]) * y_view[0])

    return x_offset, y_offset

# 去除孔洞，去除飞地
def process_mask(mask):
    # 确保掩膜是uint8类型
    mask = mask.astype(np.uint8) * 255

    # 使用闭运算去除孔洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 连通组件分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    # 保留最大的连通区域（假设最大区域是我们想要保留的区域）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 跳过背景标签
    mask = np.where(labels == largest_label, 1, 0).astype(np.uint8)

    return mask

def process_Automask(mask):
    # 使用形态学闭操作去除孔洞
    kernel_close = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)

    # 使用形态学开操作平滑边缘
    kernel_open = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)

    # 可选：使用高斯模糊或中值滤波平滑边缘
    smoothed = cv2.GaussianBlur(opening, (5, 5), 0)

    # 提取轮廓并填充
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(smoothed)
    cv2.drawContours(filled_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return filled_mask

# 切换到点击模式的函数
def switch_to_click_mode():
    global mode
    mode = "click"

# 切换到拖拽模式的函数
def switch_to_drag_mode():
    global mode
    mode = "drag"

# 切换到拖拽模式的函数
def switch_to_mask_mode():
    global mode
    mode = "mask"

def on_click(event):
    global image1, last_image, image_history, update_mask
    global mode
    print(f"Clicked! Current mode: {mode}")  # 调试信息
    if mode == "drag":
        return
    # 在处理图像之前，保存当前状态
    image_history.append(update_mask.copy())

    x_offset, y_offset = get_scroll_offset()

    # 调整点击坐标为Canvas上的实际坐标
    x = event.x + x_offset
    y = event.y + y_offset

    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks1, scores, logits = masks2.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # 将得分和掩膜绑定在一起并按得分排序
    scored_masks = sorted(zip(scores, masks1), key=lambda x: x[0], reverse=True)

    # 遍历排序后的掩膜列表，选择合适的掩膜
    selected_mask = None
    for score, mask in scored_masks:
        if np.sum(mask) < 50000:
            selected_mask = mask
            max_score = score
            break

    # 如果没有找到合适的掩膜，保持原样
    if selected_mask is None:
        print("No suitable mask found")
        return

    # 后处理掩膜
    masks = process_mask(selected_mask.astype(np.uint8))

    if np.sum(masks) < 50000:  # overlap_rate < 0.8 and
        uint8_matrix = (masks.astype(np.uint8) * 255)
        kernel = np.ones((15, 15), np.uint8)  # 3x3 方形结构元素

        # 膨胀操作
        dilated_m = cv2.dilate(uint8_matrix, kernel, iterations=1)
        dilated_m = cv2.erode(dilated_m, kernel, iterations=1)
        masks = dilated_m.astype(bool)
        color = np.random.random(3)
        h, w = masks.shape[-2:]
        maskscol = masks.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        update_mask = update_mask + maskscol

        # 生成带有 Alpha 通道的图像
        alpha_channel = np.where(update_mask[:, :, 0], 255, 0).astype(np.uint8)  # 黑色区域透明（0），其余不透明（255）
        rgba_image = np.concatenate([update_mask, alpha_channel[..., None]], axis=-1)

        mask_pil = Image.fromarray(rgba_image.astype(np.uint8))
        mask_tk = ImageTk.PhotoImage(mask_pil)

        # 如果之前有图像，则删除它
        if last_image is not None:
            canvas.delete(last_image)

        # 创建新的图像并保存引用
        last_image = canvas.create_image(0, 0, anchor="nw", image=mask_tk)

        # 重要：保存对mask_tk的引用，防止被垃圾收集
        mask_tk.image = mask_tk

        layer_images["Mask"] = Image.fromarray(update_mask.astype(np.uint8))
        if "Mask" not in layer_order:
            layer_order.append("Mask")

        # 更新图层选择下拉列表
        update_layer_combobox()

        print("Clicked at coordinates: ({}, {}), and scores is: {}".format(x, y, scores))

def on_right_click(event):
    global image1, last_image, image_history, update_mask

    # 检查是否有先前的图像状态可供恢复
    if image_history:
        # 恢复到上一状态
        update_mask = image_history.pop()

        alpha_channel = np.where(update_mask[:, :, 0], 255, 0).astype(np.uint8)  # 黑色区域透明（0），其余不透明（255）
        rgba_image = np.concatenate([update_mask, alpha_channel[..., None]], axis=-1)
        # 更新画布上的图像
        mask_pil = Image.fromarray(rgba_image.astype(np.uint8))
        mask_tk = ImageTk.PhotoImage(mask_pil)

        # 如果之前有图像，则删除它
        if last_image is not None:
            canvas.delete(last_image)

        # 创建新的图像并保存引用
        last_image = canvas.create_image(0, 0, anchor="nw", image=mask_tk)
        mask_tk.image = mask_tk

        layer_images["Mask"] = Image.fromarray(update_mask.astype(np.uint8))
        # 更新图层选择下拉列表
        update_layer_combobox()

def on_mouse_move(event):
    global mode
    if mode == "click":
        return
    global end_x, end_y
    if dragging:
        x_offset, y_offset = get_scroll_offset()
        end_x = event.x + x_offset
        end_y = event.y + y_offset
        redraw_canvas()

def on_mouse_up(event):
    global mode
    if mode == "click":
        return
    global dragging
    if dragging:
        x_offset, y_offset = get_scroll_offset()
        end_x = event.x + x_offset
        end_y = event.y + y_offset
        dragging = False
        process_selection()

def redraw_canvas():
    # 清除画布并重新绘制图像和选区
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=photo)
    if dragging:
        canvas.create_rectangle(start_x, start_y, end_x, end_y, outline="red")

def process_selection():
    global image1, last_image, image_history, update_mask
    # 在处理图像之前，保存当前状态
    image_history.append(update_mask.copy())
    # 处理选区，调用predict函数
    # 确保起始坐标小于结束坐标

    x1, y1, x2, y2 = min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)
    box = [x1, y1, x2, y2]

    box = np.array(box).reshape(1, 4)
    masks, scores, _ = masks2.predict(box=box)
    # 将得分和掩膜绑定在一起并按得分排序
    scored_masks = sorted(zip(scores, masks), key=lambda x: x[0], reverse=True)

    # 遍历排序后的掩膜列表，选择合适的掩膜
    selected_mask = None
    max_score = -1
    for score, mask in scored_masks:
        # 找出掩膜中非零的坐标
        nonzero_y, nonzero_x = np.nonzero(mask)

        # 检查掩膜是否完全在框内
        if (nonzero_x.min() >= x1) and (nonzero_x.max() <= x2) and \
            (nonzero_y.min() >= y1) and (nonzero_y.max() <= y2):
            # 检查得分
            if score > max_score:
                selected_mask = mask
                max_score = score

    # 如果没有找到合适的掩膜，则选择得分最高的掩膜
    if selected_mask is None:
        selected_mask = scored_masks[0][1] if scored_masks else None

    # 后处理掩膜
    masks = process_mask(selected_mask.astype(np.uint8)).astype(bool)

    color = np.random.random(3)
    h, w = masks.shape[-2:]
    maskscol = masks.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
    masks = masks.reshape(h, w, 1)

    # 更新遮罩

    update_mask = update_mask + maskscol

    # 生成带有 Alpha 通道的图像
    alpha_channel = np.where(update_mask[:, :, 0], 255, 0).astype(np.uint8)  # 黑色区域透明（0），其余不透明（255）
    rgba_image = np.concatenate([update_mask, alpha_channel[..., None]], axis=-1)

    mask_pil = Image.fromarray(rgba_image.astype(np.uint8))
    mask_tk = ImageTk.PhotoImage(mask_pil)

    # 如果之前有图像，则删除它
    if last_image is not None:
        canvas.delete(last_image)

    # 创建新的图像并保存引用
    last_image = canvas.create_image(0, 0, anchor="nw", image=mask_tk)

    # 重要：保存对mask_tk的引用，防止被垃圾收集
    mask_tk.image = mask_tk

    layer_images["Mask"] = Image.fromarray(update_mask.astype(np.uint8))
    if "Mask" not in layer_order:
        layer_order.append("Mask")
    # 更新图层选择下拉列表
    update_layer_combobox()

    print("Clicked at  scores is: {}".format(scores))

# 绘制多边形
def draw_polygon():
    canvas.delete("polygon")  # 删除旧的多边形
    if len(polygon_points) > 1:
        # 添加最后一个点回到第一个点，以闭合多边形
        canvas.create_line(polygon_points + [polygon_points[0]], fill="blue", width=2, tags="polygon")

# 创建多边形掩膜
def create_polygon_mask():
    mask = np.zeros((image1.shape[0], image1.shape[1]), dtype=np.uint8)
    if len(polygon_points) > 2:
        cv2.fillPoly(mask, [np.array(polygon_points)], color=(255))
    return mask

# 完成多边形
def finish_polygon():
    global polygon_points, layer_images, layer_order, update_canvas, update_layer_combobox, mask_manual
    mask_manual = create_polygon_mask()

    # 创建独立的掩膜图层
    mask_image = Image.fromarray(mask_manual)  # 生成灰度图
    layer_name = f"Polygon Mask {len(layer_images) + 1}"
    layer_images[layer_name] = mask_image

    # 将新图层置顶
    layer_order.remove(layer_name) if layer_name in layer_order else None
    layer_order.insert(0, layer_name)

    # 重置多边形顶点列表
    polygon_points = []

    # 更新画布和图层选择下拉列表
    update_canvas()
    update_layer_combobox()

def handle_mouse_click(event):
    global mode, start_x, start_y, dragging, polygon_points
    if mode == "click":
        # 点击模式的处理逻辑
        on_click(event)
        print(f"Clicked! Current mode: {mode}")
    elif mode == "drag":
        # 拖拽模式的开始处理逻辑
        x_offset, y_offset = get_scroll_offset()
        start_x = event.x + x_offset
        start_y = event.y + y_offset
        dragging = True
    elif mode == "mask":
        # 拖拽模式的开始处理逻辑
        # 点击模式的处理逻辑
        x_offset, y_offset = get_scroll_offset()
        x = event.x + x_offset
        y = event.y + y_offset
        polygon_points.append((x, y))  # 添加点到多边形顶点列表
        draw_polygon()  # 绘制多边形
        print(f"Clicked! Current mode: {mode}")

def save_updated_part():
    file_path = filedialog.asksaveasfilename(defaultextension=".tif")
    if file_path:
        global image1, update_mask

        # 使用GDAL创建新的数据集
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(file_path, update_mask.shape[1], update_mask.shape[0], 1, gdal.GDT_Byte)
        out_band = out_dataset.GetRasterBand(1)

        # 将掩膜数据写入新数据集
        out_band.WriteArray(update_mask[:, :, 0])  # 假设掩膜存储在update_mask的第一个通道

        # 应用地理参考和投影信息
        out_dataset.SetGeoTransform(geo_transform)
        out_dataset.SetProjection(projection)

        # 关闭数据集
        out_band.FlushCache()
        out_dataset = None

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
    pointX = points[:, 0] * w
    pointY = points[:, 1] * h
    point_image = np.array([pointX, pointY])
    point_image = point_image.transpose((1, 0)).astype(np.int16)
    mask_points = [mask[coord[0], coord[1]] for coord in point_image]
    mask_points = np.array(mask_points).reshape(-1, 1)
    filtered_matrix = points * mask_points
    filtered_matrix_zeros = filtered_matrix[np.logical_not(np.all(filtered_matrix == [0, 0], axis=1))]
    filtered_matrix_zeros = np.fliplr(filtered_matrix_zeros)
    pointX = filtered_matrix_zeros[:, 0] * w
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

def SEA_auto():
    # 定义模型：
    global update_mask, mask_manual, sam_checkpoint, numstrid
    device = "cuda"  # 使用GPU
    sam = sam_model_registry[selected_model_type](checkpoint=selected_sam_checkpoint)
    sam.to(device=device)
    crop_n_points_downscale_factor = 1
    stride = [numstrid]
    Total_Seed = []
    point_number = 3000
    if mask_manual is not None:  # 如果 mask_manual 已经被定义（不是 None）
        DW = mask_manual // 255
    else:
        DW = np.ones((image1.shape[:2]))
    for i in stride:
        print(i)
        Total_Seed.append(define_area_average_Field(DW, i))
        # Total_Seed.append(random_points_from_matrix(DW, point_number))

    masks2 = SamPredictor(sam)
    masks2.set_image(image1)
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
        selected_mask = process_Automask(selected_mask.astype(np.uint8))
        # 使用形态学闭操作去除孔洞
        kernel = np.ones((9, 9), np.uint8)
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
                update_mask[current_mask] = random_color
                count_num += 1
    # 添加图像到图层字典
    layer_images["Mask"] = Image.fromarray(update_mask)
    if "Mask" not in layer_order:
        layer_order.append("Mask")

    # 更新图层选择下拉列表
    update_layer_combobox()

# 绑定事件时，添加一些交互效果
def on_enter(e):
    try:
        e.widget['background'] = bg_color
    except tk.TclError:
        # 如果widget不支持背景色更改，则忽略错误
        pass

def on_leave(e):
    try:
        e.widget['background'] = bg_color
    except tk.TclError:
        # 如果widget不支持背景色更改，则忽略错误
        pass

def resource_path(relative_path):
    """获取资源的绝对路径。用于PyInstaller打包后资源的访问。"""
    try:
        # PyInstaller创建的临时文件夹
        base_path = sys._MEIPASS
    except Exception:
        # 正常的Python环境
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

icon_path = resource_path("icon/favicon.ico")

root = ThemedTk(theme="breeze")  # 使用breeze主题，您可以选择其他主题
root.iconbitmap(icon_path)
root.geometry("1200x600")
root.title("STAMP")

# 使用统一的配色方案和字体
bg_color = "#f0f0f0"
fg_color = "#333333"
font_name = "Helvetica"
font_size = 10

# 设置应用的默认字体
root.option_add("*Font", f"{font_name} {font_size}")
root.configure(bg=bg_color)

# 创建一个顶级菜单
menubar = tk.Menu(root, bd=1, relief=tk.RAISED, bg=bg_color)
root.config(menu=menubar)

# 创建菜单项
file_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=open_image)
file_menu.add_command(label="Save", command=save_updated_part)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

# 创建说明选项
about_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="About", menu=about_menu)
about_menu.add_command(label="Help", command=open_image)
about_menu.add_separator()
about_menu.add_command(label="Licence", command=save_updated_part)

# 创建工具栏样式
style = ttk.Style()
style.configure("Custom.TFrame", background=bg_color, borderwidth=1)

# 创建工具栏
toolbar = ttk.Frame(root, relief=tk.RAISED, style="Custom.TFrame")
toolbar.pack(side=tk.TOP, fill=tk.X)

# 创建按钮样式
style.configure("TButton", padding=1, relief="flat", background=bg_color)

# 添加模型选择标签
model_select_label = tk.Label(toolbar, text="Model Selected:")
model_select_label.pack(side=tk.LEFT, padx=2)

model_combobox = ttk.Combobox(toolbar, values=sam_checkpoints)
model_combobox.current(0)  # 设置默认选择
model_combobox.pack(side=tk.LEFT, padx=5)

# 分隔符
separator1 = ttk.Separator(toolbar, orient='vertical')
separator1.pack(side=tk.LEFT, fill='y', padx=5)

# 向工具栏添加按钮
open_button = ttk.Button(toolbar, text="Open", compound=tk.LEFT, style="TButton", command=open_image)
open_button.pack(side=tk.LEFT, padx=2, pady=2)

save_button = ttk.Button(toolbar, text="Save", compound=tk.LEFT, style="TButton", command=save_updated_part)
save_button.pack(side=tk.LEFT, padx=2, pady=2)

# 创建清除数据按钮
clear_button = ttk.Button(toolbar, text="Clear All", compound=tk.LEFT, style="TButton", command=clear_all_data)
clear_button.pack(side=tk.LEFT, padx=2, pady=2)

# 创建工具栏
toolbar2 = ttk.Frame(root, relief=tk.RAISED, style="Custom.TFrame")
toolbar2.pack(side=tk.TOP, fill=tk.X)
# 添加手动模式标签
Autolabel = tk.Label(toolbar2, text="Manual mode:")
Autolabel.pack(side=tk.LEFT, padx=2)

Click_button = ttk.Button(toolbar2, text="Click", compound=tk.LEFT, style="TButton", command=switch_to_click_mode)
Click_button.pack(side=tk.LEFT, padx=2, pady=2)
Drag_button = ttk.Button(toolbar2, text="Drag", compound=tk.LEFT, style="TButton", command=switch_to_drag_mode)
Drag_button.pack(side=tk.LEFT, padx=2, pady=2)

# 创建工具栏
toolbar1 = ttk.Frame(root, relief=tk.RAISED, style="Custom.TFrame")
toolbar1.pack(side=tk.TOP, fill=tk.X)
# 添加自动模式标签
Autolabel = tk.Label(toolbar1, text="Auto Model:")
Autolabel.pack(side=tk.LEFT, padx=2)

AutoSeg_button = ttk.Button(toolbar1, text="STAMP", compound=tk.LEFT, style="TButton", command=SEA_auto)
AutoSeg_button.pack(side=tk.LEFT, padx=2, pady=2)

Mask_button = ttk.Button(toolbar1, text="Mask", compound=tk.LEFT, style="TButton", command=switch_to_mask_mode)
Mask_button.pack(side=tk.LEFT, padx=2, pady=2)

# 添加输入框
numstrid_label = tk.Label(toolbar1, text="SampleNumber:")
numstrid_label.pack(side=tk.LEFT, padx=2)

numstrid_entry = tk.Entry(toolbar1)
numstrid_entry.pack(side=tk.LEFT, padx=2)
numstrid_entry.insert(0, "64")  # 设置默认值
# 添加更新按钮
update_numstrid_button = ttk.Button(toolbar1, text="Update Numstrid", command=update_numstrid)
update_numstrid_button.pack(side=tk.LEFT, padx=2)

# 创建操作区域
operation_frame = tk.Frame(root, bg=bg_color)
operation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 创建图层选择界面
layers_frame = tk.Frame(operation_frame, bd=2, relief=tk.GROOVE, padx=5, pady=5)
layers_frame.pack(side=tk.TOP, fill=tk.X)

layer_var = tk.StringVar()
layer_label = tk.Label(layers_frame, text="Choose a layer:")
layer_label.pack(side=tk.LEFT, padx=5)
layer_combobox = ttk.Combobox(layers_frame, textvariable=layer_var)
# layer_combobox['values'] = ("Layer 1", "Layer 2", "Layer 3")
layer_combobox.pack(side=tk.LEFT, padx=5)

layer_combobox.bind("<<ComboboxSelected>>", on_select)

# 创建画布区域
canvas_frame = tk.Frame(operation_frame)
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame, bg="white")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 创建滚动条
scroll_y = tk.Scrollbar(canvas, orient="vertical", command=canvas.yview)
scroll_y.pack(side=tk.RIGHT, fill="y")
scroll_x = tk.Scrollbar(canvas, orient="horizontal", command=canvas.xview)
scroll_x.pack(side=tk.BOTTOM, fill="x")

# 初始化透明度滑块
opacity_scale = Scale(layers_frame, from_=0, to=100, orient=tk.HORIZONTAL)
opacity_scale.pack(side=tk.LEFT)

# 配置画布滚动
canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

open_button.bind("<Enter>", on_enter)
open_button.bind("<Leave>", on_leave)

# 绑定鼠标左键点击事件
canvas.bind("<Button-1>", handle_mouse_click)

# 绑定鼠标右键点击事件
canvas.bind("<Button-3>", on_right_click)

# 绑定鼠标拖拽事件
# canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)

# 绑定透明度滑块的值变化事件
opacity_scale.config(command=update_layer_opacity)

# 绑定双击事件以完成多边形
canvas.bind("<Double-1>", lambda event: finish_polygon())

# 绑定模型超参数的选择
model_combobox.bind("<<ComboboxSelected>>", update_model_selection)

root.mainloop()





