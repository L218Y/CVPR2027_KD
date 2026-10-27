import torch
import torch.nn as nn
from torch import Tensor

from semseg.models.cmnext import CMNeXt


class EarlyFusionCMNeXt(nn.Module):
    """Single CMNeXt-B2 with six-channel (RGB+thermal) early fusion.

    The thermal stream is concatenated at the input, so the deployed model has
    the same parameter count and FLOPs as the RGB-only expert.  The first
    patch-embedding convolution is expanded after RGB pre-training; its thermal
    half is initialized from the RGB filters.
    """

    def __init__(self, backbone='CMNeXt-B2', num_classes=9, modals=None):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.net = CMNeXt(backbone, num_classes, ['img'])
        self._expand_first_conv()

    def _expand_first_conv(self, source_weight=None, source_bias=None):
        old = self.net.backbone.patch_embed1.proj
        new = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride,
                        old.padding, bias=old.bias is not None)
        with torch.no_grad():
            if source_weight is None:
                source_weight = old.weight
                source_bias = old.bias
            new.weight[:, :3] = source_weight
            new.weight[:, 3:] = source_weight * 0.5
            if source_bias is not None and new.bias is not None:
                new.bias.copy_(source_bias)
        self.net.backbone.patch_embed1.proj = new

    def init_pretrained(self, pretrained=None):
        if not pretrained:
            return
        tmp = CMNeXt(self.backbone_name, self.num_classes, ['img'])
        tmp.init_pretrained(pretrained)
        src = tmp.backbone.patch_embed1.proj
        sd = tmp.state_dict()
        sd.pop('backbone.patch_embed1.proj.weight', None)
        sd.pop('backbone.patch_embed1.proj.bias', None)
        self.net.load_state_dict(sd, strict=False)
        self._expand_first_conv(src.weight, src.bias)

    def forward(self, x) -> Tensor:
        img = torch.cat([x[0], x[1]], dim=1) if isinstance(x, (list, tuple)) else x
        return self.net([img])
