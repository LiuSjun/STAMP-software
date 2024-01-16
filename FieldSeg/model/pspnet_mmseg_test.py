from mmcv.utils import Config
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models import build_segmentor
from mmseg.models.builder import SEGMENTORS

MODEL_PATH = r"D:\CropSegmentation\data\mmseg_data\file\PSPNet_config.py"

# @SEGMENTORS.register_module()
# class EncoderDecoderSelf(EncoderDecoder):
#     def __init__(self,
#                  backbone,
#                  decode_head,
#                  neck=None,
#                  auxiliary_head=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  pretrained=None,
#                  init_cfg=None):
#         super(EncoderDecoderSelf, self).__init__(backbone, decode_head, neck=neck,
#                 auxiliary_head=auxiliary_head, train_cfg=train_cfg, test_cfg=test_cfg,
#                 pretrained=pretrained, init_cfg=init_cfg)
#
#     def forward(self, img):
#         x = self.extract_feat(img)
#         out_head = self._decode_head_forward_train(x)
#         out_auxiliary_head = self._decode_head_forward_train(x)
#         return out_head, out_auxiliary_head


def PSPNet():
    cfg = Config.fromfile(MODEL_PATH)
    cfg_model = cfg.model
    model = build_segmentor(
        cfg_model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    return model
