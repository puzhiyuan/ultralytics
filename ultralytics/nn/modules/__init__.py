# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    ImagePoolingAttn,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
    ContrastiveHead,
    BNContrastiveHead,
    RepNCSPELAN4,
    ADown,
    SPPELAN,
    CBFuse,
    CBLinear,
    Silence,
    # TODO add Module
    LSKAttention,
    LSKAttentionV2,
    SPPFCSPC,
    Concat_BiFPN,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

from .swTransformer import (
    SwinTransformer,
    C3STR,
)

from .CoreV8.Backbone.emo import C3_RMB, CSRMBC, C2f_RMB, CPNRMB, ReNLANRMB
from .CoreV8.Backbone.biformer import CSCBiF, ReNLANBiF, CPNBiF, C3_Biformer, C2f_Biformer
from .CoreV8.Backbone.CFNet import CSCFocalNeXt, ReNLANFocalNeXt, CPNFocalNeXt, C3_FocalNeXt, C2f_FocalNeXt
from .CoreV8.Backbone.FasterNet import FasterNeXt, CSCFasterNeXt, ReNLANFasterNeXt, C3_FasterNeXt, C2f_FasterNeXt
from .CoreV8.Backbone.Ghost import CPNGhost, CSCGhost, ReNLANGhost, C3_Ghost, C2f_Ghost
from .CoreV8.Backbone.EfficientRep import RepVGGBlock, SimConv, RepBlock, Transpose
from .CoreV8.Backbone.damo import CReToNeXt
from .CoreV8.Backbone.QARep import QARep, CSCQARep, ReNLANQARep, C3_QARep, C2f_QARep
from .CoreV8.Backbone.ConvNeXtv2 import CPNConvNeXtv2, CSCConvNeXtv2, ReNLANConvNeXtv2, C3_ConvNeXtv2, C2f_ConvNeXtv2
from .CoreV8.Backbone.MobileViTv1 import CPNMobileViTB, CSCMobileViTB, ReNLANMobileViTB, C3_MobileViTB, C2f_MobileViTB
from .CoreV8.Backbone.MobileViTv2 import CPNMVBv2, CSCMVBv2, ReNLANMVBv2, C3_MVBv2, C2f_MVBv2
from .CoreV8.Backbone.MobileViTv3 import CPNMViTBv3, CSCMViTBv3, ReNLANMViTBv3, C3_MViTBv3, C2f_MViTBv3
from .CoreV8.Backbone.RepLKNet import CPNRepLKBlock, CSCRepLKBlock, ReNLANRepLKBlock, C3_RepLKBlock, C2f_RepLKBlock

from .CoreV8.Neck.GELAN import RepNCSPELAN4
from .CoreV8.Neck.AFPN import ASFF_2, ASFF_3, BasicBlock
from .CoreV8.Neck.GDM import (
    LAF_px, low_FAM, LAF_h, low_IFM, InjectionMultiSum_Auto_pool1,
    InjectionMultiSum_Auto_pool2, InjectionMultiSum_Auto_pool3, InjectionMultiSum_Auto_pool4,
    PyramidPoolAgg, TopBasicLayer)

from .CoreV8.Neck.SSFF import SSFF
from .CoreV8.SPPF.SimSPPF import SimConv, SimSPPF
from .CoreV8.SPPF.ASPP import ASPP
from .CoreV8.SPPF.BasicRFB import BasicRFB

from .Improve.Attention.simam import SimAM
from .Improve.Attention.sk import SKAttention
from .Improve.Attention.soca import SOCA
from ultralytics.nn.modules.new.seam import SEAM, MultiSEAM
from .CoreV8.Impove.CARAFE import CARAFE
from .new.ECANet import ECA
from .new.CoTattention import CoT
from .new.GAMNet import GAMAttention
from .new.SANet import ShuffleAttention
from .new.CBAMNet import CBAM, Res_CBAM
from .new.backbone.MSBlock import MS

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    # TODO add Module
    "LSKAttention",
    "LSKAttentionV2",
    "SPPFCSPC",
    "SwinTransformer",
    "C3STR",
    "Concat_BiFPN",
    # -------------------
    'C3_RMB', 'CSRMBC', 'C2f_RMB', 'CPNRMB', 'ReNLANRMB',
    'CSCBiF', 'ReNLANBiF', 'CPNBiF', 'C3_Biformer', 'C2f_Biformer',
    'CSCFocalNeXt', 'ReNLANFocalNeXt', 'CPNFocalNeXt', 'C3_FocalNeXt', 'C2f_FocalNeXt',
    'FasterNeXt', 'CSCFasterNeXt', 'ReNLANFasterNeXt', 'C3_FasterNeXt', 'C2f_FasterNeXt',
    'CPNGhost', 'CSCGhost', 'ReNLANGhost', 'C3_Ghost', 'C2f_Ghost',
    'RepVGGBlock', 'SimConv', 'RepBlock', 'Transpose',
    'CReToNeXt',
    'QARep', 'CSCQARep', 'ReNLANQARep', 'C3_QARep', 'C2f_QARep',
    'CPNConvNeXtv2', 'CSCConvNeXtv2', 'ReNLANConvNeXtv2', 'C3_ConvNeXtv2', 'C2f_ConvNeXtv2',
    'CPNMobileViTB', 'CSCMobileViTB', 'ReNLANMobileViTB', 'C3_MobileViTB', 'C2f_MobileViTB',
    'CPNMVBv2', 'CSCMVBv2', 'ReNLANMVBv2', 'C3_MVBv2', 'C2f_MVBv2',
    'CPNMViTBv3', 'CSCMViTBv3', 'ReNLANMViTBv3', 'C3_MViTBv3', 'C2f_MViTBv3',
    'CPNRepLKBlock', 'CSCRepLKBlock', 'ReNLANRepLKBlock', 'C3_RepLKBlock', 'C2f_RepLKBlock',
    'RepNCSPELAN4',
    'ASFF_2', 'ASFF_3', 'BasicBlock',
    'LAF_px', 'low_FAM', 'LAF_h', 'low_IFM', 'InjectionMultiSum_Auto_pool1',
    'InjectionMultiSum_Auto_pool2', 'InjectionMultiSum_Auto_pool3', 'InjectionMultiSum_Auto_pool4',
    'PyramidPoolAgg', 'TopBasicLayer',
    'SSFF',
    'SimConv', 'SimSPPF',
    'ASPP',
    'BasicRFB',
    'SimAM',
    'GAMAttention',
    'CBAM',
    'SKAttention',
    'SOCA',
    'ShuffleAttention',
    'CARAFE',
    'SEAM',
    'MultiSEAM',
    'ECA',
    'CoT',
    'Res_CBAM',
    'MS',
)
