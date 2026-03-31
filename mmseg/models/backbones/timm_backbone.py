import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, trunc_normal_

from mmseg.models.builder import BACKBONES, HEADS, build_loss
from mmseg.ops import resize


@BACKBONES.register_module()
class TIMMBackbone(BaseModule):
    """Wrapper around timm models that produce multi-scale feature maps.

    Args:
        model_name (str): timm model name.
        features_only (bool): Return feature maps from intermediate layers.
        pretrained (bool): Load pretrained weights.
        out_indices (tuple): Indices of feature maps to return.
        init_cfg (dict): Init config.
    """

    def __init__(self,
                 model_name='resnet50.fb_swsl_ig1b_ft_in1k',
                 features_only=True,
                 pretrained=True,
                 out_indices=(1, 2, 3, 4),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.model = timm.create_model(
            model_name,
            features_only=features_only,
            pretrained=pretrained,
            out_indices=out_indices)
        self.out_indices = out_indices

    def forward(self, x):
        return self.model(x)

    def init_weights(self):
        pass


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6())


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels))


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2))


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6())


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self, dim=256, num_heads=16, qkv_bias=False,
                 window_size=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, 1, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ac = nn.Conv2d(dim, dim, 1, 1, groups=dim)
        self.pa = nn.Identity()
        self.act = nn.ReLU6()

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.num_heads
        head_dim = C // N

        x = self.ac(x) * x
        x = self.pa(x)
        x = self.act(x)

        # Window-based attention
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[2], x.shape[3]

        x = x.reshape(B, N, head_dim, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 3, 5, 2, 4, 6, 1).reshape(B * (Hp // ws) * (Wp // ws), N, ws * ws, head_dim)

        qkv = self.qkv(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], qkv.shape[2], 3, self.num_heads, head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(qkv.shape[0], qkv.shape[1], qkv.shape[2], -1)
        x = self.proj(x)
        x = x.reshape(B, Hp // ws, Wp // ws, ws, ws, N, head_dim).permute(0, 5, 1, 3, 2, 4, 6).reshape(B, N, Hp, Wp)

        if pad_h > 0:
            x = x[:, :, :H, :]
        if pad_w > 0:
            x = x[:, :, :, :W]

        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                          window_size=window_size, attn_drop=attn_drop,
                                          proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                       act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32) * self.eps)

    def forward(self, x, res):
        x = self.pre_conv(x)
        fuse = self.weights[0] * x + self.weights[1] * res
        return fuse


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.pa = ConvBNReLU(decode_channels, decode_channels)
        self.ca = ConvBNReLU(decode_channels, decode_channels)
        self.proj = Conv(decode_channels, decode_channels, kernel_size=1)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = self.pre_conv(x)
        shortcut = self.proj(x)
        x = self.pa(x)
        x = self.ca(x * res)
        x = shortcut + x
        x = self.act(x)
        return x


class AuxHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


@HEADS.register_module()
class UNetFormerDecoderHead(BaseModule):
    """UNetFormer Decoder adapted for MMSeg v0.16.0.

    Wraps the reference Decoder with the standard BaseDecodeHead interface
    so it works with EncoderDecoder segmentor.

    Args:
        encoder_channels (tuple): Channel dims from backbone feature maps.
        decode_channels (int): Decoder hidden channels.
        num_classes (int): Number of segmentation classes.
        dropout (float): Dropout rate.
        window_size (int): Window attention size.
        loss_decode (dict): Loss config for segmentation.
        init_cfg (dict): Init config.
    """

    def __init__(self,
                 encoder_channels=(256, 512, 1024, 2048),
                 decode_channels=256,
                 num_classes=9,
                 dropout=0.1,
                 window_size=8,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 ignore_index=255,
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.encoder_channels = encoder_channels

        # Classification layers (mimics BaseDecodeHead)
        self.conv_seg = nn.Conv2d(decode_channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # UNetFormer Decoder blocks
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.seg_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        res1, res2, res3, res4 = inputs[0], inputs[1], inputs[2], inputs[3]

        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.p1(x, res1)

        return x

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, seg_weight=None):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        feat = self.forward(inputs)
        seg_logits = self.seg_head(feat)
        h, w = img_metas[0]['img_shape'][:2] if img_metas else feat.shape[2:]
        seg_logits = F.interpolate(seg_logits, size=(h, w), mode='bilinear', align_corners=False)
        return seg_logits

    def losses(self, seg_logit, seg_label, seg_weight=None):
        loss = dict()
        seg_logit = resize(
            input=seg_logit, size=seg_label.shape[2:],
            mode='bilinear', align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = F.cross_entropy(
            seg_logit, seg_label.long(),
            ignore_index=self.ignore_index, reduction='mean')
        return loss
