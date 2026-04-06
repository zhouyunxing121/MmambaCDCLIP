print(">>> DEBUG: clip_backbone.py is being imported!")
from collections import OrderedDict
from typing import Tuple, Union
import math
from timm.models.layers import trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import drop_path, trunc_normal_
from mmseg.registry import MODELS
from mmseg.models.backbones import ResNet
from mmseg.models.backbones import VisionTransformer as MMVisionTransformer

from timm.models.resnet import ResNet as TimmResNet
from timm.models.resnet import Bottleneck as TimmBottleneck
import timm

import math
from timm.models.vision_transformer import VisionTransformer
try:
    from mamba_ssm import Mamba as VMambaBlock  
except ImportError:
    print("Mamba not installed. Run: pip install mamba-ssm")
    VMambaBlock = None


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        #位置编码插值
        cls_pos = self.positional_embedding[0:1, :]
        spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)#位置编码

        x = x + positional_embedding[:, None, :]#加位置编码
        x, _ = F.multi_head_attention_forward(#多头自注意力
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map

@MODELS.register_module()
class CLIPResNet(nn.Module):#没有用到
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        return tuple(outs)

@MODELS.register_module()
class CLIPResNetWithAttention(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, 32, output_dim)
        # self.init_weights()

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 32
                            old_h = int(math.sqrt(state_dict[new_k][1:,].shape[0]))
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, old_h, old_h, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        x_global, x_local = self.attnpool(x)
        outs.append([x_global, x_local])

        return tuple(outs)





#没有用到
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dt_rank="auto", dropout=0.1, bias=False, dtype=None, device=None):
        super().__init__()
        self.dim = dim
        self.mamba = VMambaBlock(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            bias=bias,
            dropout=dropout,# 注意：mamba_ssm 的 Mamba 可能不支持 dropout！
            dtype=dtype,#查看 mamba_ssm 官方 API，其 Mamba 层 不直接接受 device/dtype，而是跟随输入张量。
            #所以这些参数可能是多余的，甚至导致错误。
            device=device#查看 mamba_ssm 官方 API，其 Mamba 层 不直接接受 device/dtype，而是跟随输入张量。
            #所以这些参数可能是多余的，甚至导致错误。
        )

    def forward(self, x):
        """
        x: (B, H*W, C) or (B, C, H, W)
        returns: (B, H*W, C)
        """
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, -1).permute(0, 2, 1)  # -> (B, H*W, C)

        x = self.mamba(x)  # applies over sequence length

        return x
"""
def expand_4dir_to_8dir_state_dict(state_dict, strict=True):
    import warnings
    DIR_KEYS = ['A_logs', 'dt_projs_weight', 'dt_projs_bias', 'Ds']
    new_state_dict = {}
    for k, v in state_dict.items():
        expanded = False
        if v.dim() == 0:
            new_state_dict[k] = v
            continue
        for dir_key in DIR_KEYS:
            if dir_key in k:
                if dir_key == 'Ds':
                    if v.numel() % 4 == 0:
                        d_inner = v.numel() // 4
                        v8 = torch.cat([v.view(4, d_inner), v.view(4, d_inner)], dim=0).view(-1)
                        new_state_dict[k] = v8
                        expanded = True
                        break
                elif v.shape[0] == 4:
                    new_state_dict[k] = torch.cat([v, v], dim=0)
                    expanded = True
                    break
                elif v.shape[0] == 8:
                    new_state_dict[k] = v
                    expanded = True
                    break
        if not expanded:
            new_state_dict[k] = v
    return new_state_dict
"""
"""
def expand_4dir_to_8dir_state_dict(state_dict, strict=True):
    import torch
    # 所有与方向数（k_group）相关的参数名关键词
    DIR_KEYS = [
        'A_logs',
        'dt_projs_weight',
        'dt_projs_bias',
        'Ds',
        #'x_proj.weight',      
        'x_proj_weight',
    ]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if v.dim() == 0:
            new_state_dict[k] = v
            continue

        expanded = False
        for dir_key in DIR_KEYS:
            if dir_key in k:
                # 处理 x_proj.weight：shape [4 * proj_dim, d_inner] → 需要按方向拆分
                #if 'x_proj.weight' in k:
                if 'x_proj.weight' in k:
                    # 假设 proj_dim = (dt_rank + d_state * 2)，总 dim = 4 * proj_dim
                    # 我们简单地将前 1/4、2/4... 视为每个方向的投影
                    # 更安全的方式：确认 proj_dim 后 reshape
                    total_dim = v.shape[0]
                    if total_dim % 4 == 0:
                        proj_dim = total_dim // 4
                        # 拆成 4 个方向，再复制一次变成 8 个
                        v4 = v.view(4, proj_dim, -1)  # [4, proj_dim, d_inner]
                        v8 = torch.cat([v4, v4], dim=0).view(-1, v.shape[1])  # [8*proj_dim, d_inner]
                        new_state_dict[k] = v8
                        expanded = True
                        print(f"[INFO] Expanded {k} from {v.shape} to {v8.shape}")
                    else:
                        new_state_dict[k] = v  # fallback
                        print(f"[WARN] Cannot expand {k}: shape {v.shape} not divisible by 4")
                    break

                # 其他参数：第一维是 k_group（4 → 8）
                elif v.shape[0] == 4:
                    v8 = torch.cat([v, v], dim=0)
                    new_state_dict[k] = v8
                    expanded = True
                    print(f"[INFO] Expanded {k} from {v.shape} to {v8.shape}")
                    break
                elif v.shape[0] == 8:
                    new_state_dict[k] = v
                    expanded = True
                    break

        if not expanded:
            new_state_dict[k] = v

    return new_state_dict

"""
def expand_4dir_to_8dir_state_dict(state_dict, strict=True):
    import torch
    # 注意：checkpoint 中的 key 是 x_proj_weight，不是 x_proj.weight！
    DIR_KEYS = [
        'A_logs',
        'dt_projs_weight',
        'dt_projs_bias',
        'Ds',
        'x_proj_weight',   # 修正：使用实际 key 名称
    ]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if v.dim() == 0:
            new_state_dict[k] = v
            continue

        expanded = False
        for dir_key in DIR_KEYS:
            if dir_key in k:
                # 所有方向相关参数：第一维是 k_group (4 → 8)
                if v.shape[0] == 4:
                    v8 = torch.cat([v, v], dim=0)  # 复制一份，变成 8
                    new_state_dict[k] = v8
                    expanded = True
                    print(f"[INFO] Expanded {k} from {v.shape} to {v8.shape}")
                    break
                elif v.shape[0] == 8:
                    new_state_dict[k] = v
                    expanded = True
                    print(f"[INFO] Kept {k} as is: {v.shape}")
                    break
                else:
                    # 更安全的方式：检查是否能被 4 整除，并且是 1D 或 2D
                    if dir_key in ['A_logs', 'Ds'] and len(v.shape) in [1, 2]:
                        total = v.shape[0]
                        if total % 4 == 0:
                            chunk_size = total // 4
                            chunks = torch.chunk(v, 4, dim=0)  # 分成 4 份
                            v8 = torch.cat(chunks + chunks, dim=0)  # 复制 → 8 份
                            new_state_dict[k] = v8
                            print(f"[INFO] Expanded {k} (chunked): {v.shape} → {v8.shape}")
                            expanded = True
                            break
                    
                    # 可能是其他层（如 layer0 的 k_group=4 但 shape 不同）
                    # 保守起见，不扩展
                    new_state_dict[k] = v
                    print(f"[WARN] Unexpected shape for {k}: {v.shape}, not expanded")
                    expanded = True
                    break

        if not expanded:
            new_state_dict[k] = v

    return new_state_dict


from .vmamba import VSSM, PatchEmbed
@MODELS.register_module()
class CLIPMambaWithAttention(nn.Module):

    def __init__(
        self,
        # Mamba Encoder 参数
        patch_size=4,
        #patch_size=3,
        
        in_chans=3,
        #in_chans=6,
        num_classes=0,  
        
        #depths=[2, 2, 9, 2],
        #depths=[2,2,27,2],
        depths=[2, 2, 15, 2],
        embed_dim=768,#embed_dim = dims[-1] 
        #dims=[48, 96, 192, 384],
        dims=[96, 192, 384, 768],
        #dims=[128,256,512,1024],
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_dt_rank="auto",
        #disable_z=False,
        ssm_disable_z: bool = False,
        ssm_drop_rate=0.0,
        #drop_path_rate=0.1,
        drop_path_rate=0.3,
        use_checkpoint=False,
        # CLIP Head 参数
        output_dim=768,
        input_resolution=224,
        pretrained_mamba=None,
        patchembed_version="v2",
        **kwargs
    ):
        super().__init__()
        self.pretrained = pretrained_mamba
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.dims = dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.ssm_dt_rank = ssm_dt_rank
        print(">>> Received dims:", dims)
        
        # === Mamba Backbone  ===
        self.mamba_encoder = VSSM(
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # 不分类
            depths=depths,
            dims=dims,
            #d_state=ssm_d_state,
            ssm_d_state=ssm_d_state,
            ssm_drop_rate=ssm_drop_rate,
            ssm_ratio=ssm_ratio,
            ssm_conv=ssm_conv,
            ssm_dt_rank=ssm_dt_rank,
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint,
            downsample_version="v1",  
            patchembed_version=patchembed_version,
            #disable_z=disable_z,
            ssm_disable_z=ssm_disable_z,
        )
        
       
        # === 移除原始 VSSM 的分类头 ===
        del self.mamba_encoder.classifier
        #del self.mamba_encoder.norm
        if hasattr(self.mamba_encoder, 'norm'):
            del self.mamba_encoder.norm
        else:
            print(f"Warning: mamba_encoder has no attribute 'norm'. Available attributes: {dir(self.mamba_encoder)}")

        # 要提取各阶段的中间特征
        # 根据 ChangeMamba 的 VSSM 实现，forward_features 返回 list of features
        # 即 [stage0, stage1, stage2, stage3] 输出

        # === Attention Pooling Head ===
        """
        self.spatial_size = input_resolution // patch_size  # e.g., 224//4=56
        embed_dim = dims[-1]  # 768

        self.attnpool = AttentionPool2d(
            self.spatial_size, embed_dim, num_heads=embed_dim // 64, output_dim=output_dim
        )
        """
        # 计算最后一 stage 的 spatial size
        h = w = input_resolution
        h = h // patch_size  # stage0
        for _ in range(len(depths) - 1):  # stage1,2,3 共3次下采样
            h = h // 2
        final_spatial_size = h  # e.g., 224 → 74 → 37 → 18 → 9

        self.attnpool = AttentionPool2d(
            final_spatial_size, dims[-1], num_heads=dims[-1] // 64, output_dim=output_dim
        )
        # 初始化
        self.apply(self._init_weights)
        """
        if pretrained_mamba:
            self.init_weights(pretrained_mamba)
        """
        """
        if pretrained_mamba:
            self.init_weights(pretrained_mamba)
        else:
        # 从头训练：初始化 mamba_encoder
            self.mamba_encoder.apply(self._init_weights_vssm)
            print(f"[INFO] Training from scratch. Initialized mamba_encoder with dims={self.dims}.")
        """
      
    """
    def _init_weights_vssm(self, m):
        
        #初始化 VSSM 子模块。
        
        if isinstance(m, nn.Conv2d):
        # PatchEmbed / PatchMerging 中的 Conv
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    """
    def _init_weights_vssm(self, m):
    # 跳过 SSM 特有参数（由 SS2D 自己初始化）
        if hasattr(m, "_no_weight_decay") or (hasattr(m, "weight") and getattr(m.weight, "_no_reinit", False)):
            return

        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    #def init_weights(self, pretrained):
    def init_weights(self):
        """加载 ChangeMamba 预训练权重，并支持 3→6 通道扩展"""
        if self.pretrained is None:
        # 从头训练：只初始化 mamba_encoder
            print("pretrained is None.Use _init_weights_vssm to initialize mamba_encoder")
            self.mamba_encoder.apply(self._init_weights_vssm)
            print(f"[INFO] Training from scratch. Initialized mamba_encoder with dims={self.dims}.")
            return
        # 在 clip_backbone.py 的 init_weights 开头加：
        print("Model dims:", self.dims)
        for i, layer in enumerate(self.mamba_encoder.layers):
            d_model = layer.blocks[0].op.in_proj.weight.shape[1]
            print(f"Stage {i} d_model: {d_model}")
        pretrained=self.pretrained
        print(f"Loading pretrained Mamba weights from {pretrained}")
        ckpt = torch.load(pretrained, map_location='cpu')
    
        # 提取 state_dict
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # 移除分类头相关参数
        state_dict = {
            k: v for k, v in state_dict.items() 
            if not k.startswith('classifier') and 'head' not in k and not k.startswith('norm')
        }
        """
       # === Step 2: 手动重命名 key ===
        new_sd = {}
        for k, v in state_dict.items():
            if k == "patch_embed.proj.weight":
                new_sd["patch_embed.weight"] = v
            elif k == "patch_embed.proj.bias":
                new_sd["patch_embed.bias"] = v
            elif ".ln_1." in k:
                new_sd[k.replace(".ln_1.", ".norm.")] = v
            elif ".self_attention." in k:
                new_sd[k.replace(".self_attention.", ".op.")] = v
            else:
                new_sd[k] = v
        state_dict = new_sd
        """
        # =============== STEP 1: 扩展 4方向 → 8方向 ===============
        state_dict = expand_4dir_to_8dir_state_dict(state_dict, strict=False)

        
        # === 扩展输入通道（3 → 6）===
        if self.in_chans == 6:
            print("self.in_chans == 6,实现3通道输入扩展为6通道")
            expanded = False
            if 'patch_embed.weight' in state_dict:
                weight = state_dict['patch_embed.weight']  # [out, in, k, k]
                if weight.shape[1] == 3 and self.in_chans == 6:
                    state_dict['patch_embed.weight'] = torch.cat([weight, weight], dim=1)
                    print(f"[INFO] Expanded patch_embed.weight from 3 to 6 channels.")
                    expanded = True
            if 'patch_embed.0.weight' in state_dict:
                weight = state_dict['patch_embed.0.weight']  # [out, in=3, k, k]
                if weight.shape[1] == 3 and self.in_chans == 6:
                    state_dict['patch_embed.0.weight'] = torch.cat([weight, weight], dim=1)
                    print(f"[INFO] Expanded patch_embed.0.weight from 3 to 6 channels.")
                    expanded = True
            #if not expanded and self.mamba_encoder.in_chans == 6:
            if not expanded and self.in_chans == 6:
                print("[WARNING] No patch_embed weight found for expansion! Check your checkpoint.")
    
        else:
            pass
        """
        expanded = False
        if 'patch_embed.weight' in state_dict:
            weight = state_dict['patch_embed.weight']  # [out, in, k, k]
            if weight.shape[1] == 3 and self.in_chans == 6:
                state_dict['patch_embed.weight'] = torch.cat([weight, weight], dim=1)
                print(f"[INFO] Expanded patch_embed.weight from 3 to 6 channels.")
                expanded = True
        if 'patch_embed.0.weight' in state_dict:
            weight = state_dict['patch_embed.0.weight']  # [out, in=3, k, k]
            if weight.shape[1] == 3 and self.in_chans == 6:
                state_dict['patch_embed.0.weight'] = torch.cat([weight, weight], dim=1)
                print(f"[INFO] Expanded patch_embed.0.weight from 3 to 6 channels.")
                expanded = True
        #if not expanded and self.mamba_encoder.in_chans == 6:
        if not expanded and self.in_chans == 6:
            print("[WARNING] No patch_embed weight found for expansion! Check your checkpoint.")
        """
        # === 添加 mamba_encoder 前缀 ===
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[f'mamba_encoder.{k}'] = v

        # 加载权重
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)
        
        
    def forward(self, x):
        B, C, H, W = x.shape  # e.g., (B, 6, 224, 224)

        features = self.mamba_encoder.forward_features(x)  # List of (B, L, D) if channel_first=False

        outs = []
        current_h, current_w = H, W

        for i, feat in enumerate(features):
            if i == 0:
            # 第一个 stage：patch_embed 下采样 patch_size 倍
                current_h = current_h // self.patch_size  # 224 // 3 = 74
                current_w = current_w // self.patch_size  # 74
            else:
                # 后续 stage：每个 PatchMerging 下采样 2 倍
                current_h = current_h // 2
                current_w = current_w // 2

            if feat.dim() == 3:  # (B, L, D)
                L = feat.size(1)
                expected_L = current_h * current_w
                assert L == expected_L, f"Stage {i}: L={L}, expected {expected_L} (H'={current_h}, W'={current_w})"
                D = self.dims[i]
                feat = feat.transpose(1, 2).view(B, D, current_h, current_w)

            outs.append(feat)

        # Attention pooling on last feature
        x_global, x_local = self.attnpool(outs[-1])
        outs.append([x_global, x_local])

        return tuple(outs)


'''
import clip  
@MODELS.register_module()
#class PureCLIP(BaseModule):
class PureCLIP(nn.Module):
    """
    纯CLIP视觉编码器,
    直接加载官方CLIP的视觉模型,保留原类的所有接口和输出结构
    """
    def __init__(self, layers=None, output_dim=1024, input_resolution=224, width=64, 
                 pretrained="ViT-B/32", init_cfg=None, **kwargs):
        # 兼容原类的初始化参数，layers/width等参数仅做兼容，不使用
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        
        # 加载官方CLIP视觉编码器（核心：纯CLIP）
        # 支持：ViT-B/32, ViT-B/16, ViT-L/14 等官方CLIP模型
        self.clip_model, _ = clip.load(pretrained, device="cpu", jit=False)
        self.visual = self.clip_model.visual  # 提取CLIP视觉主干
        
        # 强制对齐输出维度（与原类保持一致）
        if self.visual.output_dim != self.output_dim:
            self.proj = nn.Linear(self.visual.output_dim, self.output_dim)
        else:
            self.proj = nn.Identity()

    def init_weights(self, pretrained=None):
        """权重初始化，兼容原类接口"""
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            # 官方CLIP已预训练，无需额外加载，直接使用
            print(f"Loaded pretrained PureCLIP from official checkpoint: {pretrained}")
            # 固定CLIP权重（可选，根据需求修改）
            for param in self.visual.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        前向传播：输出格式与原类完全一致
        输出：tuple(
            layer1_out, layer2_out, layer3_out, layer4_out, [global_feat, local_feat]
        )
        """
        # 兼容原类的dtype转换
        x = x.type(self.visual.conv1.weight.dtype)
        
        # ========== 模拟原类的4层输出（CLIP ViT无层级，用特征分阶段输出兼容） ==========
        outs = []
        # CLIP ViT 前向传播
        x = self.visual(x)  # CLIP输出全局特征 [B, output_dim]
        
        # 构造与原类一致的4层中间输出（占位，保证shape/格式兼容）
        B, C, H, W = 1, 64, self.input_resolution//4, self.input_resolution//4
        dummy_feat = torch.randn(B, C, H, W).to(x.device).type(x.dtype)
        outs.append(dummy_feat)  # layer1
        outs.append(dummy_feat)  # layer2
        outs.append(dummy_feat)  # layer3
        outs.append(dummy_feat)  # layer4

        # ========== 全局+局部特征（与原类输出格式一致） ==========
        x_global = self.proj(x)  # 对齐输出维度
        # CLIP无显式局部特征，用全局特征reshape模拟（保证输出结构一致）
        x_local = x_global.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        
        outs.append([x_global, x_local])

        # 返回元组，与原类输出格式100%一致
        return tuple(outs)
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from mmengine.model import BaseModule  # 必须用MMEngine的BaseModule

import torch
import torch.nn as nn
from mmengine.registry import MODELS

# ---------------------------------------------
# 🔥 完全独立的原生 CLIP Visual (仅视觉部分)
# 不依赖你项目里的任何类！不依赖环境clip！
# 就是你要的：只是 CLIP！
# ---------------------------------------------
class NativeCLIPVisual(nn.Module):
    def __init__(self, embed_dim=768, input_resolution=512, patch_size=14, width=768, layers=24, heads=12):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution//patch_size)**2 +1, width))
        self.ln_pre = nn.LayerNorm(width)
        
        self.transformer = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=width, 
                nhead=heads, 
                dim_feedforward=width*4,
                activation='gelu',
                batch_first=False
            ) for _ in range(layers)
        ])
        
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, embed_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2).permute(0,2,1)
        x = torch.cat([self.class_embedding.unsqueeze(0).expand(x.shape[0],-1,-1), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1,0,2)
        x = self.transformer(x)
        x = x.permute(1,0,2)
        x = self.ln_post(x)[:,0,:] @ self.proj
        return x

# ---------------------------------------------
# ✅ 你要的 PureCLIP：只是 CLIP！
# ---------------------------------------------
@MODELS.register_module()
class PureCLIP(nn.Module):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__()
        # 🔥 完全独立的CLIP，不碰你项目里的任何代码
        self.visual = NativeCLIPVisual()
        self.pretrained = pretrained

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if pretrained:
            sd = torch.load(pretrained, map_location="cpu")
            visual_sd = {k.replace("visual.",""):v for k,v in sd.items() if k.startswith("visual.")}
            self.visual.load_state_dict(visual_sd, strict=False)

    def forward(self, x):
        return self.visual(x)


@MODELS.register_module()
class TimmResnet50(nn.Module):


    def __init__(self, layers, output_dim=1024, input_resolution=224, width=64, pretrained=None, **kwargs):
        super().__init__()
        self.extracter = timm.create_model('resnet50', pretrained=True, features_only=True)
        
    def init_weights(self, pretrained=None):
        print('using resnet50')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = self.extracter(x)
        return tuple(outs)


class LayerNorm(nn.LayerNorm):
    

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x

@MODELS.register_module()
class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512, drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None, get_embeddings=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.GroupNorm(1, embed_dim)

            self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.GroupNorm(1, embed_dim)

            self.fpn3 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        elif patch_size == 32:
            self.fpn1 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Sequential(
                nn.GroupNorm(1, embed_dim),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn4 = nn.GroupNorm(1, embed_dim)

        
    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size*self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]


        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if i in self.out_indices:
                xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                features.append(xp.contiguous())

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
        
        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W

            features.append([global_embedding, visual_embedding])

        return tuple(features)
   



@MODELS.register_module()
class CLIPTextEncoder(nn.Module):
    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.init_weights()

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = self.out_proj(x)
        return x


@MODELS.register_module()
class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=22,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        def init_weights(self, pretrained=None):
            pretrained = pretrained or self.pretrained
            if isinstance(pretrained, str):
                print(f"[INFO] Loading text encoder weights from: {pretrained}")
                # 1. 加载完整的 TorchScript 模型并获取其 state_dict
                full_state_dict = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

                # 2. 构建一个干净的、与当前模型结构匹配的 state_dict
                state_dict = {}

                for k, v in full_state_dict.items():
                    # --- 处理文本编码器部分 ---
                    # 在 TorchScript 模型中，文本编码器的键通常以 'text_encoder.' 开头
                    if k.startswith('text_encoder.'):
                        # 移除 'text_encoder.' 前缀
                        new_k = k[len('text_encoder.'):]
                    
                        # 将 transformer 层的权重直接映射
                        if new_k.startswith('transformer.'):
                            state_dict[new_k] = v
                    
                        # 处理其他顶层文本编码器参数
                        elif new_k in ['positional_embedding', 'text_projection']:
                            # 截断 positional_embedding 以适应自定义的 context_length
                            if new_k == 'positional_embedding' and v.size(0) > self.context_length:
                                v = v[:self.context_length]
                                print(f'[INFO] Truncated positional_embedding from {v.size(0)} to {self.context_length}')
                            state_dict[new_k] = v
                    
                        # 处理 token_embedding 和 ln_final
                        elif new_k.startswith('token_embedding') or new_k.startswith('ln_final'):
                            state_dict[new_k] = v
                
                    # --- 兜底：为了兼容性，也尝试处理不带前缀的旧格式 ---
                    # (这通常不会触发，但保留以增加鲁棒性)
                    else:
                        if k.startswith('transformer.'):
                            state_dict[k] = v
                        if k in ['positional_embedding', 'text_projection'] or k.startswith('token_embedding') or k.startswith('ln_final'):
                            if k == 'positional_embedding' and v.size(0) > self.context_length:
                                v = v[:self.context_length]
                                print(f'[INFO] Truncated positional_embedding from {v.size(0)} to {self.context_length}')
                            state_dict[k] = v

                # 3. 加载权重
                missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
                if missing_keys:
                    print(f"[WARNING] Missing keys in text encoder: {missing_keys}")
                if unexpected_keys:
                    print(f"[WARNING] Unexpected keys in text encoder: {unexpected_keys}")
            
                print("[INFO] Text encoder initialization completed.")
    """
    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
             
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')
    """

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context):


        x_text = self.token_embedding(text)  # n_clas, n_text, C
        N, K, N1, C = x_text.shape
        N, B, N2, C = context.shape

        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(N, K).unsqueeze(1).reshape(-1)

        # x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
        context = context.expand(N, K, N2, C)

        x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(N*K, N1+N2, C)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(N, K, self.embed_dim)
        return x


@MODELS.register_module()
class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)
        
        return self.out_proj(x)
