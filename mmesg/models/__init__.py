# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, build_backbone,
                      build_head, build_loss, build_segmentor)
from .data_preprocessor import SegDataPreProcessor
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
#F401 和 F403 是 flake8 中的错误代码，分别表示不同的问题。
#F401：表示模块被导入但未使用。
#如果你导入了 os 模块但没有在代码中使用它，flake8 会报出 F401 警告。通过添加 # noqa: F401，你可以告诉 flake8 忽略这一警告。
#F403：表示使用了 from module import *

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor', 'SegDataPreProcessor'
]
