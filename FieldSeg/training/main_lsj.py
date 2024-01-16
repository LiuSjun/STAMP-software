from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import torch.nn as nn
import os, torch, UNIT
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
# from model.UNet import UNet, DeepLabV3, PAN
from model.resnet_unet_0903 import UNet_Bou
from TanimotoLoss import TanimotoLoss, LovaszSoftmaxV1, FocalLoss
from albumentations import (
    RandomRotate90, Transpose, Flip,  Compose
)
from osgeo import gdal
from torch.utils.tensorboard import SummaryWriter
import cv2

dataset_root = r"F:\Code\CropPatch\Crop\share\data"

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D
def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D
def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=True):
        super(CannyFilter, self).__init__()

        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)
        self.gaussian_filter = self.gaussian_filter.cuda()

        # sobel
        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)
        self.sobel_filter_x = self.sobel_filter_x.cuda()

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)
        self.sobel_filter_y = self.sobel_filter_y.cuda()
        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)
        self.directional_filter = self.directional_filter.cuda()
        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)
        self.hysteresis = self.hysteresis.cuda()


    def forward(self, img, low_threshold=100, high_threshold=200, hysteresis=True):
        # set the setps tensors
        img = img
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian
        for c in range(C):
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

        # thick edges
        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds
        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1
#blurred, grad_x, grad_y, grad_magnitude, grad_orientation,thin_edges
        return thin_edges


def augmentation(**images):
    '''
    通过图像左右、上下翻转进行增强
    Returns:
    '''
    band_size = [0, ]
    images_concatenate = []
    for key in images:
        temp = np.transpose(images[key], (1, 2, 0))
        band_size.append(temp.shape[2] + band_size[-1])
        images_concatenate.append(temp)
    images_concatenate = np.concatenate(images_concatenate, axis=2)

    compose = Compose([RandomRotate90(p=0.5), Flip(p=0.5)], p=1)
    images_concatenate = compose(image=images_concatenate)["image"]

    for i, key in enumerate(images):
        temp = images_concatenate[:, :, band_size[i]: band_size[i+1]]
        temp = np.transpose(temp, (2, 0, 1))
        images[key] = temp
    return images


def normalize_parameters(img):
    '''
    获取影像各个波段的normalize parameters
    '''
    # top = np.percentile(img, 98, axis=(1, 2))
    # bottom = np.percentile(img, 2, axis=(1, 2))
    # return top, bottom

    listpoint =[0,0,0,0]
    aaa = img.reshape((4,-1))
    aaa0 = aaa[0]
    bbb0 = np.where(aaa[0]!= listpoint[0], False, True)
    ccc0 = aaa0[[~bbb0]]
    top0 = np.percentile(ccc0, 98)
    bottom0 = np.percentile(ccc0, 2)

    aaa1 = aaa[1]
    bbb1 = np.where(aaa[1]!= listpoint[1], False, True)
    ccc1 = aaa1[[~bbb1]]
    top1 = np.percentile(ccc1, 98)
    bottom1 = np.percentile(ccc1, 2)

    aaa2 = aaa[2]
    bbb2 = np.where(aaa[2]!= listpoint[2], False, True)
    ccc2 = aaa2[[~bbb2]]
    top2 = np.percentile(ccc2, 98)
    bottom2 = np.percentile(ccc2, 2,)

    aaa3 = aaa[3]
    bbb3 = np.where(aaa[3]!= listpoint[3], False, True)
    ccc3 = aaa3[[~bbb3]]
    top3 = np.percentile(ccc3, 98)
    bottom3 = np.percentile(ccc3, 2)

    top = [top0,top1,top2,top3]
    bottom = [bottom0, bottom1, bottom2, bottom3]

    return top, bottom


def normalize_apply(img, paras):
    para_top, para_bottom = paras
    for i in range(len(para_top)):
        img_bnd = img[i, :, :]
        top, bottom = para_top[i], para_bottom[i]

        img_bnd[img_bnd > top] = top
        img_bnd[img_bnd < bottom] = bottom
        img_bnd = (img_bnd - bottom) / (top - bottom) * 255
        img[i, :, :] = img_bnd
    img = img.astype("uint8")
    return img


def canny_process(img, th1=100, th2=200):
    lenimg = img.shape[0]
    for i in range(lenimg):
        img1 = img[i]
        img[i] = cv2.Canny(img1, th1, th2, L2gradient=True)
    return img//255


def sober_process(img):
    lenimg = img.shape[0]
    for i in range(lenimg):
        img1 = img[i]
        sobelx1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        sobelx1 = cv2.convertScaleAbs(sobelx1)

        # 加入绝对值后Gy效果
        sobely1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        sobely1 = cv2.convertScaleAbs(sobely1)

        # 二者求和
        img[i]  = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)
    return img//255


def preprocessing(image,
                  mean_coeff=(479.32578151, 643.47553845, 624.71096795, 1575.66875469),
                  std_coeff=(369.24628162, 481.19982643, 498.34453097, 1243.93270332)):
    '''
    根据系数，将图像白化
    Returns:
    '''
    # for i in range(4):
    #     # image[i, :, :] = (image[i, :, :].astype(np.float32) - mean_coeff[i]) / std_coeff[i]
    #     image[i, :, :] = (image[i, :, :] - mean_coeff[i]) / std_coeff[i]
    return image


#更改结构，只进行分割
from torch.utils.data import random_split


class DatasetSegtask(BaseDataset):
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
            images_dir,
            segmentation_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.images_names = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, name) for name in self.images_names]
        self.segmentation_fps = [os.path.join(segmentation_dir, name) for name in self.images_names]

        # convert str names to class values on masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # print(self.images_fps[i])
        # read data
        image = UNIT.img2numpy(self.images_fps[i]).astype(np.float32)

        segmentation = UNIT.img2numpy(self.segmentation_fps[i])
        # segmentation = np.where(segmentation == 0, 0, 1).astype(np.uint8)
        segmentation = np.expand_dims(segmentation, axis=0)

        # visualize(image=image, mask=mask)
        # apply augmentations
        # visualize_aug(image, distance, boundary, segmentation)
        if self.augmentation:
            data = self.augmentation(image=image,
                                     seg=segmentation)
            image, segmentation= data["image"], data["seg"]
        # apply preprocessing
        # visualize_aug(image, distance, boundary, segmentation)
        if self.preprocessing:
            image = self.preprocessing(image=image)
        return image, segmentation

    def __len__(self):
        return len(self.images_fps)


class DatasetBouSegtask(BaseDataset):
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
            images_dir,
            segmentation_dir,
            boundary_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.images_names = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, name) for name in self.images_names]
        self.segmentation_fps = [os.path.join(segmentation_dir, name) for name in self.images_names]
        self.boundary_fps = [os.path.join(boundary_dir, name) for name in self.images_names]

        # convert str names to class values on masks
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # print(self.images_fps[i])
        # read data
        image = UNIT.img2numpy(self.images_fps[i]).astype(np.float32)

        segmentation = UNIT.img2numpy(self.segmentation_fps[i])
        boundary = UNIT.img2numpy(self.boundary_fps[i])
        # segmentation = np.where(segmentation == 0, 0, 1).astype(np.uint8)
        segmentation = np.expand_dims(segmentation, axis=0)
        boundary = np.expand_dims(boundary, axis=0)

        # visualize(image=image, mask=mask)
        # apply augmentations
        # visualize_aug(image, distance, boundary, segmentation)
        if self.augmentation:
            data = self.augmentation(image=image,
                                     seg=segmentation,
                                     bou=boundary)
            image, segmentation,boundary = data["image"], data["seg"], data["bou"]
        # apply preprocessing
        # visualize_aug(image, distance, boundary, segmentation)
        if self.preprocessing:
            image = self.preprocessing(image=image)
        return image, segmentation, boundary

    def __len__(self):
        return len(self.images_fps)
val_percent = 0.1


def training_seg_boun():
    def acc_metric(predb, yb):
        predb = (predb > 0.5).float()
        acc = (predb == yb).float().mean()
        return acc

    def f1_metric(predb, yb):
        predb = (predb > 0.5).float()

        tp = (yb * predb).sum().to(torch.float32)
        tn = ((1 - yb) * (1 - predb)).sum().to(torch.float32)
        fp = ((1 - yb) * predb).sum().to(torch.float32)
        fn = (yb * (1 - predb)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return f1
    # model_vgg = lanenet.VGG()
    # model_vgg = torch.nn.DataParallel(model_vgg).cuda()

    spe_dir = os.path.join(dataset_root, "Spec") #Spectral Spec
    # canny_dir = os.path.join(dataset_root, "Canny")
    segmentation_dir = os.path.join(dataset_root, "Semantic")   #  Boundary Semantic Label
    dataset_extent = DatasetSegtask(spe_dir, segmentation_dir,  augmentation=augmentation,  preprocessing=preprocessing)
    n_val = int(len(dataset_extent) * val_percent)
    n_train = len(dataset_extent) - n_val
    train_dataset_extent, val_dataset_extent = random_split(dataset_extent, [n_train, n_val])

    model = UNet_Bou().cuda()
    load_name = ''
    # model_vgg.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_name:
        model.load_state_dict(
            torch.load(load_name, map_location=device)
        )

    loss_fn = LovaszSoftmaxV1() #TanimotoLoss()LovaszSoftmaxV1 BatchSoftDiceLoss  FocalLoss # loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    print("Data Loading...")
    trainloader = torch.utils.data.DataLoader(train_dataset_extent, batch_size=36, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset_extent, batch_size=36, shuffle=True, num_workers=0)
    writer = SummaryWriter(comment=f'LR_{0.001}_BS_{255}', log_dir="writ")
    global_step = 0
    print("Loading Finished.")
    with torch.autograd.set_detect_anomaly(True):
        train_epoch_best_loss = 100.0
        for epoch in range(500):
            loss_total = 0
            for step, data in enumerate(trainloader, 0):
                image, seg = data
                image = image.to(torch.float32).cuda()
                seg = seg.to(torch.int64).cuda()
                # seg = seg.cuda()

                optimizer.zero_grad()
                out_seg = model(image)

                loss_seg = loss_fn(out_seg, seg)

                # x_vgg = model_vgg(out_seg)
                # y_vgg = model_vgg(seg)
                # loss_vgg = loss_fn(x_vgg, y_vgg)
                loss = loss_seg
                loss.backward()
                loss_total += loss.item()

                optimizer.step()
                if step % 20 == 0:
                    print('Current epoch-step: {}-{}  '
                          '<<<Loss>>>: {}  '
                          'AllocMem (Mb): {}'.
                          format(epoch, step,  loss, torch.cuda.memory_allocated() / 1024 / 1024))
                    print("Detail Loss:<<SEG>>:{}:".
                          format(loss_seg))
                    print()
            loss_total = loss_total/ len(trainloader)
            writer.add_scalars('train/test', {'train_loss': loss_total}, global_step)
            scheduler.step()
            if epoch % 1 == 0:
                # torch.save(model.state_dict(), "model_state_e{}.pth".format(epoch))
                print("---Validation e {}: ".format(epoch), end=", ")
                num = 0
                f_score = 0
                acc = 0
                loss_val = 0
                with torch.no_grad():
                    for step, data in enumerate(valloader, 0):
                        image,seg = data
                        image = image.to(torch.float32).cuda()
                        seg = seg.to(torch.int64).cuda()
                        out_seg = model(image)
                        loss_seg = loss_fn(out_seg, seg)
                        loss_val += loss_seg
                        f_score += f1_metric(out_seg, seg)
                        acc += acc_metric(out_seg, seg)
                        num += 1
                print("\033[31m"
                      "F score: ",
                      f_score/num, "; Accuracy: ",
                      acc/num, "LR: ", scheduler.get_last_lr(),
                      ".\033[0m")
                loss_val = loss_val / len(valloader)
                # writer.add_scalars('train/test', {'test_loss': loss_val}, global_step)
            train_epoch_loss = loss_val
            if train_epoch_loss >= train_epoch_best_loss:
                no_optim += 1
            else:
                no_optim = 0
                train_epoch_best_loss = train_epoch_loss
                torch.save(model.state_dict(), "model_state_{}_Seg.pth".format(epoch))
            global_step += 1
    writer.close()


def training_seg_boun_canny():
    def acc_metric(predb, yb):
        predb = (predb > 0.5).float()
        acc = (predb == yb).float().mean()
        return acc

    def f1_metric(predb, yb):
        predb = (predb > 0.5).float()

        tp = (yb * predb).sum().to(torch.float32)
        tn = ((1 - yb) * (1 - predb)).sum().to(torch.float32)
        fp = ((1 - yb) * predb).sum().to(torch.float32)
        fn = (yb * (1 - predb)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return f1
    # model_vgg = lanenet.VGG()
    # model_vgg = torch.nn.DataParallel(model_vgg).cuda()

    spe_dir = os.path.join(dataset_root, "Spec")
    # canny_dir = os.path.join(dataset_root, "Canny")
    segmentation_dir = os.path.join(dataset_root, "Semantic")   #  Boundary Semantic
    boundary_dir = os.path.join(dataset_root, "Boundary")  # Boundary Semantic
    dataset_extent = DatasetBouSegtask(spe_dir, segmentation_dir,boundary_dir, augmentation=augmentation,  preprocessing=preprocessing)
    n_val = int(len(dataset_extent) * val_percent)
    n_train = len(dataset_extent) - n_val
    train_dataset_extent, val_dataset_extent = random_split(dataset_extent, [n_train, n_val])

    model = UNet_Bou().cuda()
    load_name = ''
    # model_vgg.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_name:
        model.load_state_dict(
            torch.load(load_name, map_location=device)
        )


    loss_fn = LovaszSoftmaxV1() #TanimotoLoss()LovaszSoftmaxV1 BatchSoftDiceLoss  FocalLoss # loss_fn = nn.CrossEntropyLoss()
    edgetensor = CannyFilter()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    print("Data Loading...")
    trainloader = torch.utils.data.DataLoader(train_dataset_extent, batch_size=36, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset_extent, batch_size=36, shuffle=True, num_workers=0)
    writer = SummaryWriter(comment=f'LR_{0.001}_BS_{255}', log_dir="writ")
    global_step = 0
    print("Loading Finished.")
    with torch.autograd.set_detect_anomaly(True):
        train_epoch_best_loss = 100.0
        for epoch in range(500):
            loss_total = 0
            for step, data in enumerate(trainloader, 0):
                image, seg, ed = data
                image = image.to(torch.float32).cuda()
                seg = seg.to(torch.int64).cuda()
                ed = ed.to(torch.int64).cuda()
                # seg = seg.cuda()

                optimizer.zero_grad()
                out_seg = model(image)
                edge = edgetensor(out_seg)
                loss_edge = loss_fn(edge, ed)
                loss_seg = loss_fn(out_seg, seg)

                # x_vgg = model_vgg(out_seg)
                # y_vgg = model_vgg(seg)
                # loss_vgg = loss_fn(x_vgg, y_vgg)
                loss = loss_seg + loss_edge
                loss.backward()
                loss_total += loss.item()

                optimizer.step()
                if step % 20 == 0:
                    print('Current epoch-step: {}-{}  '
                          '<<<Loss>>>: {}  '
                          'AllocMem (Mb): {}'.
                          format(epoch, step,  loss, torch.cuda.memory_allocated() / 1024 / 1024))
                    print("Detail Loss:<<SEG>>:{}:".
                          format(loss_seg))
                    print()
            loss_total = loss_total/ len(trainloader)
            writer.add_scalars('train/test', {'train_loss': loss_total}, global_step)
            scheduler.step()
            if epoch % 1 == 0:
                # torch.save(model.state_dict(), "model_state_e{}.pth".format(epoch))
                print("---Validation e {}: ".format(epoch), end=", ")
                num = 0
                f_score = 0
                acc = 0
                loss_val = 0
                with torch.no_grad():
                    for step, data in enumerate(valloader, 0):
                        image,seg, ed = data
                        image = image.to(torch.float32).cuda()
                        seg = seg.to(torch.int64).cuda()
                        out_seg = model(image)
                        loss_seg = loss_fn(out_seg, seg)
                        loss_val += loss_seg
                        f_score += f1_metric(out_seg, seg)
                        acc += acc_metric(out_seg, seg)
                        num += 1
                print("\033[31m"
                      "F score: ",
                      f_score/num, "; Accuracy: ",
                      acc/num, "LR: ", scheduler.get_last_lr(),
                      ".\033[0m")
                loss_val = loss_val / len(valloader)
                # writer.add_scalars('train/test', {'test_loss': loss_val}, global_step)
            train_epoch_loss = loss_val
            if train_epoch_loss >= train_epoch_best_loss:
                no_optim += 1
            else:
                no_optim = 0
                train_epoch_best_loss = train_epoch_loss
                torch.save(model.state_dict(), "model_state_{}_Seg.pth".format(epoch))
            global_step += 1
    writer.close()


def inference_seg(img, Model, name, interval=200, start=16):
    '''
    将整个影像分割为各个255 * 255的小块，推测完后加回去。
    '''
    mask = None
    model = Model.cuda()
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint)

    size = 256*2
    b, xlen, ylen = img.shape
    half_size = int(size / 2)
    x_center = np.arange(start, xlen, interval, dtype=np.int)
    y_center = np.arange(start, ylen, interval, dtype=np.int)
    x_center, y_center = np.meshgrid(x_center, y_center)

    img_count = np.zeros((xlen, ylen), dtype=np.int16)
    img_seg = np.zeros((xlen, ylen), dtype=np.float32)

    xlen_chip, ylen_chip = x_center.shape
    with torch.autograd.set_detect_anomaly(True) and torch.no_grad():
        for i in tqdm(range(xlen_chip)):
            for j in range(ylen_chip):
                xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min((x_center[i, j] + half_size, xlen))
                yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min((y_center[i, j] + half_size, ylen))
                subset_img = np.zeros((b, size, size), dtype=np.float32)
                xsize, ysize = xloc1 - xloc0, yloc1 - yloc0
                subset_img[:, :xsize, :ysize] = img[:, xloc0:xloc1, yloc0:yloc1]
                # subset_img = normalize_apply(subset_img, normalize_parameters(subset_img)).astype(np.float32)
                if np.max(subset_img) == 0:
                    continue
                no_value_loc = np.where(np.mean(subset_img, axis=0) == 0, 0, 1)
                subset_img = preprocessing(subset_img)

                subset_img = np.expand_dims(subset_img, 0)
                subset_img_torch = torch.from_numpy(subset_img).cuda()
                model.eval()
                out_seg = model(subset_img_torch)  # critical

                out_seg = out_seg.cpu().numpy()
                out_seg =out_seg.astype(np.float32)

                out_seg *= no_value_loc

                if mask is not None:
                    img_count[xloc0:xloc1, yloc0:yloc1] += mask[:xsize, :ysize]
                    out_seg *= mask
                else:
                    img_count[xloc0:xloc1, yloc0:yloc1] += 1

                img_seg[xloc0:xloc1, yloc0:yloc1] += out_seg[0, 0, :xsize, :ysize]

    epsilon = 1e-7
    img_seg = img_seg / (img_count + epsilon)
    return img_seg


if __name__ == "__main__":

    training_seg_boun()

#     #######################  坐标文件的位置 ###############################
#     dts = gdal.Open(r"G:\CropPatch\GE_GROP\Google_Img\HuanTai.tif") #D:\Crop_C\MAP杯数智农业大赛地块识别初赛辅助数据\
#     proj = dts.GetProjection()
#     geot = dts.GetGeoTransform()
#     # #########################  预测数据所在文件夹的位置 ###############################
#     os.chdir(r"D:\Crop_C")
#
#     img_subset_1 = UNIT.img2numpy(r"G:\CropPatch\GE_GROP\Google_Img\HuanTai.tif")  # .format(i)
#     img_subset_1 = normalize_apply(img_subset_1, normalize_parameters(img_subset_1))
#
#     ##########################  模型参数路径 ###############################
#     img_seg = inference_seg(img_subset_1,
#                             UNet_Bou(),
#                             r"F:\Code\CropPatch\Crop\share\Checkpoint\G1\model_state_extent_Seg.pth")
#
#     ##########################  模型预测结果保存文件名 ###############################
#     UNIT.numpy2img(r"G:\CropPatch\GE_GROP\Google_Img\HuanTaiEXTENT.tif", img_seg, proj=proj, geot=geot)
#
#     ########################  模型参数路径 ###############################
#     img_segbou = inference_seg(img_subset_1,
#                                           UNet_Bou(),
#                                           r"F:\Code\CropPatch\Crop\share\Checkpoint\G1\model_state_bou_Seg.pth")
#
# #     ##########################  模型预测结果保存文件名 ###############################
#     UNIT.numpy2img(r"D:\Crop_C\sub2bou.tif", img_segbou, proj=proj, geot=geot)
