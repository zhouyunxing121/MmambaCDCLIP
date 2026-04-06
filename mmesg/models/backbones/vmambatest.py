import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

try:
    from .vmamba import SS2D
except:
    from vmamba import SS2D


# ✅ 关键：确保 CUDA context 初始化
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 创建模型并移到设备
model = SS2D(d_model=96, d_state=16, ssm_ratio=2.0, forward_type="v5").to(device)

# 输入也放到同一设备
x = torch.randn(2, 32, 32, 96, device=device)

# 前向传播
with torch.no_grad():
    y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("✅ v5 forward succeeded!")