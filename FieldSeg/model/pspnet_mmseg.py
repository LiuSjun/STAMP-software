'''
该模型是PSP Net 50层
'''

from mmcv.utils import Config
from mmseg.models import build_segmentor


def PSPNet():
    norm_cfg = dict(type='BN', requires_grad=True)
    cfg_model = dict(
        type='EncoderDecoder',
        pretrained='torchvision://resnet50',
        backbone=dict(
            type='ResNet',
            depth=50,
            in_channels=6,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True),
        decode_head=dict(
            type='PSPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole', crop_size=(769, 769), stride=(513, 513)))
    cfg_dict = dict(cfg_model=cfg_model)

    model = build_segmentor(Config(cfg_dict).cfg_model)
    model.init_weights()
    return model
