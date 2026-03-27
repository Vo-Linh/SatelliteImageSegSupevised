# Decode heads for Supervised Segmentation with Dynamic Anchor Module

from .decode_head import BaseDecodeHead
from .segformer_head import SegFormerHead
from .segformer_dapcn_head import SegFormerDAPCNHead
from .ddrnet_head import DDRNetHead, DDRNetDAPCNHead
from .pidnet_head import PIDNetHead, PIDNetDAPCNHead
from .knet_head import KNetHead, KNetDAPCNHead
from .segmenter_head import SegMenterHead, SegMenterDAPCNHead
from .unetformer_head import UNetFormerHead, UNetFormerDAPCNHead

__all__ = [
    'BaseDecodeHead',
    'SegFormerHead', 'SegFormerDAPCNHead',
    'DDRNetHead', 'DDRNetDAPCNHead',
    'PIDNetHead', 'PIDNetDAPCNHead',
    'KNetHead', 'KNetDAPCNHead',
    'SegMenterHead', 'SegMenterDAPCNHead',
    'UNetFormerHead', 'UNetFormerDAPCNHead',
]
