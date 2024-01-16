'''
HR Net 18 的配置代码
'''

from mmcv.utils import Config
from mmseg.models import build_segmentor


def HRNet():
    norm_cfg = dict(type='BN', requires_grad=True)
    cfg_model = dict(
        type='EncoderDecoder',
        pretrained='open-mmlab://msra/hrnetv2_w18',
        backbone=dict(
            type='HRNet',
            in_channels=6,  # mmseg\models\backbones
            norm_cfg=norm_cfg,
            norm_eval=False,
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(4,),
                    num_channels=(64,)),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='BASIC',
                    num_blocks=(4, 4),
                    num_channels=(18, 36)),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='BASIC',
                    num_blocks=(4, 4, 4),
                    num_channels=(18, 36, 72)),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block='BASIC',
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(18, 36, 72, 144)))),
        decode_head=dict(
            type='FCNHead',
            in_channels=[18, 36, 72, 144],
            in_index=(0, 1, 2, 3),
            channels=sum([18, 36, 72, 144]),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole'))
    cfg_dict = dict(cfg_model=cfg_model)

    model = build_segmentor(Config(cfg_dict).cfg_model)
    model.init_weights()
    return model
