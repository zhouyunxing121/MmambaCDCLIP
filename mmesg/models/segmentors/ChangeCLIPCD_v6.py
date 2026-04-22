# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import os

from mmseg.models.utils import resize
from mmseg.models import builder
from mmcv.cnn import ConvModule
from mmseg.models.utils.se_layer import SELayer_v2 as SELayer
from mmseg.models.utils.clip_func import clip_infer, init_clip

from ..utils.untils import tokenize

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

# ====================================================
# Mamba 依赖导入与层定义 (完全独立，不干扰原结构)
# ====================================================
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
#2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
class MambaLayer_1(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.dim = dim
        # 为了四向扫描，我们在通道维度上将输入分成4份，交给4个轻量级 Mamba 处理
        # 这是一种参数效率极高的做法，避免计算量爆炸
        self.mamba_h_fwd = Mamba(d_model=dim//4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_h_bwd = Mamba(d_model=dim//4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_v_fwd = Mamba(d_model=dim//4, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_v_bwd = Mamba(d_model=dim//4, d_state=d_state, d_conv=d_conv, expand=expand)
        
        # 将四份特征融合回原维度
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H=None, W=None):
        """
        为了垂直扫描，我们需要知道图像的高 H 和宽 W
        x: [B, N, C]  (sequence, where N = H * W)
        """
        B, N, C = x.shape
        
        # 默认回退：如果没有提供 H, W，假设它是正方形
        if H is None or W is None:
            H = W = int(N ** 0.5)

        # 1. 通道分割 (将特征切成 4 份，分别喂给 4 个方向，减小计算量)
        x_h_fwd, x_h_bwd, x_v_fwd, x_v_bwd = torch.chunk(x, 4, dim=-1)

        # 2. 水平正向 (Horizontal Forward: 左->右)
        out_h_fwd = self.mamba_h_fwd(x_h_fwd)

        # 3. 水平反向 (Horizontal Backward: 右->左)
        out_h_bwd = self.mamba_h_bwd(torch.flip(x_h_bwd, dims=[1]))
        out_h_bwd = torch.flip(out_h_bwd, dims=[1])

        # 4. 垂直正向 (Vertical Forward: 上->下)
        # 必须先 reshape 回二维，转置，再展平
        x_v_fwd_2d = x_v_fwd.reshape(B, H, W, -1).transpose(1, 2).reshape(B, N, -1)
        out_v_fwd_2d = self.mamba_v_fwd(x_v_fwd_2d)
        # 算完后再转置回来
        out_v_fwd = out_v_fwd_2d.reshape(B, W, H, -1).transpose(1, 2).reshape(B, N, -1)

        # 5. 垂直反向 (Vertical Backward: 下->上)
        x_v_bwd_2d = x_v_bwd.reshape(B, H, W, -1).transpose(1, 2).reshape(B, N, -1)
        x_v_bwd_2d_flip = torch.flip(x_v_bwd_2d, dims=[1])
        out_v_bwd_2d_flip = self.mamba_v_bwd(x_v_bwd_2d_flip)
        out_v_bwd_2d = torch.flip(out_v_bwd_2d_flip, dims=[1])
        out_v_bwd = out_v_bwd_2d.reshape(B, W, H, -1).transpose(1, 2).reshape(B, N, -1)

        # 6. 拼接四向特征并融合
        out = torch.cat([out_h_fwd, out_h_bwd, out_v_fwd, out_v_bwd], dim=-1)
        out = self.proj(out)

        # 7. 增强数值稳定
        out = torch.clamp(out, min=-1e3, max=1e3)  
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)  
        
        return out

    @property
    def d_model(self):
        return self.dim
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=1):
        super().__init__()
        self.dim = dim
        # 核心修改：只实例化一个 Mamba，通过改变输入序列的排列，实现四向特征的提取！
        # 这样不仅避免了通道切分导致的底层报错，还极其节省显存。
        self.mamba = Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv,  
            expand=expand,  
        )

    def forward(self, x, H=None, W=None):
        """
        x: [B, N, C]  (sequence)
        """
        B, N, C = x.shape
        # 兜底：如果没有传 H 和 W，默认按正方形处理
        if H is None or W is None:
            H = W = int(N ** 0.5)

        # ==========================================
        # 1. 水平正向 (左 -> 右)
        # ==========================================
        out_h_fwd = self.mamba(x)

        # ==========================================
        # 2. 水平反向 (右 -> 左)
        # ==========================================
        out_h_bwd = torch.flip(self.mamba(torch.flip(x, dims=[1])), dims=[1])

        # ==========================================
        # 3. 垂直正向 (上 -> 下)
        # ==========================================
        # 将 [B, N, C] 还原为 [B, H, W, C]，转置为 [B, W, H, C]，再展平为 [B, N, C]
        x_v = x.view(B, H, W, C).transpose(1, 2).reshape(B, N, C)
        out_v_fwd = self.mamba(x_v)
        # 再转置回来还原原始空间位置
        out_v_fwd = out_v_fwd.view(B, W, H, C).transpose(1, 2).reshape(B, N, C)

        # ==========================================
        # 4. 垂直反向 (下 -> 上)
        # ==========================================
        x_v_flip = torch.flip(x_v, dims=[1])
        out_v_bwd = torch.flip(self.mamba(x_v_flip), dims=[1])
        out_v_bwd = out_v_bwd.view(B, W, H, C).transpose(1, 2).reshape(B, N, C)

        # ==========================================
        # 5. 四向特征融合 (相加平均，保留最大共性)
        # ==========================================
        out = (out_h_fwd + out_h_bwd + out_v_fwd + out_v_bwd) / 4.0

        # 6. 增强数值稳定
        out = torch.clamp(out, min=-1e3, max=1e3)  
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)  
        
        return out

    @property
    def d_model(self):
        return self.dim

#2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析


@MODELS.register_module()
class ChangeCLIP(BaseSegmentor):
    def __init__(self,
                 backbone: ConfigType,
                 text_encoder: ConfigType,
                 context_decoder: ConfigType,
                 decode_head: ConfigType,
                 class_names=['remote sensing images', 'remote sensing images change area'],
                 context_length=5,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 tau=0.07,
                 identity_head=None,
                 token_embed_dim=512, text_dim=1024,
                 minus_channel = [256, 512, 1024, 2050], # 注意：如果用 Mamba 主干，请在配置文件中传入 [96, 192, 384, 770]
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 # === Mamba 外挂参数 ===
                 mamba_layers=False, 
                 mamba_d_state=16, 
                 mamba_d_conv=4, 
                 mamba_expand=2):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = '/home/wangshiying/.cache/clip/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = MODELS.build(backbone)
        self.text_encoder = MODELS.build(text_encoder)
        self.context_decoder = MODELS.build(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index
        self.minus_channel = minus_channel

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(class_names)        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        context_length = self.text_encoder.context_length - self.context_length
        self.contexts2 = nn.Parameter(torch.randn(1, 1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts2)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        
        
        self.minus_conv = nn.Sequential(ConvModule(
                    in_channels=self.minus_channel[0], out_channels=256, kernel_size=1),
                    ConvModule(in_channels=self.minus_channel[1], out_channels=256, kernel_size=1),
                    ConvModule(in_channels=self.minus_channel[2], out_channels=256, kernel_size=1),
                    ConvModule(in_channels=self.minus_channel[3], out_channels=256, kernel_size=1))
        
        self.channel_att = nn.Sequential(SELayer(768, 256), SELayer(768, 256), SELayer(768, 256), SELayer(768, 256))

        # ====================================================
        # Mamba 模块初始化 (仅在开启时介入计算图)
        # ====================================================
        self.mamba_layers = mamba_layers
        if self.mamba_layers:
            actual_feat_channels = [ch - (self.num_classes if i == self.score_concat_index else 0) for i, ch in enumerate(self.minus_channel)]
            
            self.mamba_modules_single = nn.ModuleList([
                MambaLayer(dim=d, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                for d in actual_feat_channels
            ])
            self.mamba_modules_fused = nn.ModuleList([
                MambaLayer(dim=768, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) 
                for _ in range(4)
            ])

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_identity_head(self, identity_head):
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        x = self.backbone(inputs)
        return x

    # ====================================================
    # 空间/序列转换工具函数 (为 Mamba 服务)
    # ====================================================
    def spatial_to_sequence(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x

    def sequence_to_spatial(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        x_g = xA[-1][0]+xB[-1][0]
        x_l = xA[-1][1]+xB[-1][1]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])
        textA, textB = self.get_cls_text(batch_img_metas, False)
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]

        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in range(len(x_clipA))]
        score_map_diff = score_mapA-score_mapB

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        
        # ====================================================
        #Mamba 融合扫描插入点 
        # ====================================================
        if getattr(self, 'mamba_layers', False):
            x_fused = []
            for i, feat in enumerate(x):
                seq_feat = self.spatial_to_sequence(feat)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                #seq_out = self.mamba_modules_single[i](seq_feat) # 换个变量名 seq_out 接收输出
                h, w = feat.shape[2], feat.shape[3]
                seq_out = self.mamba_modules_fused[i](seq_feat, H=h, W=w)
                #feat_out = self.sequence_to_spatial(seq_out, h, w)
                feat_out = self.sequence_to_spatial(seq_out, h, w)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                
                #加上残差连接
                x_fused.append(feat + feat_out)
            x = x_fused
            
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, batch_img_metas,
                                                      self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_train_with_text(self, x, textA, textB: List[Tensor],
                                            data_samples: SampleList) -> dict:
        losses = dict()
        loss_decode = self.decode_head.loss_changeclip(x, textA, textB, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, data_samples, loss_id):
        losses = dict()
        loss_aux = self.identity_head.loss(
            x, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_aux, loss_id))
        return losses

    def forward_dummy(self, img):
        seg_logit = self.encode_decode(img, None)
        return seg_logit

    def after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map
    
    def after_extract_feat_cat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map
    #2026-4-16-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
    def after_extract_feat_clip(self, x, text):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([
                global_feat.reshape(B, C, 1), 
                visual_embeddings.reshape(B, C, H*W)
            ], dim=2).permute(0, 2, 1)  # B, N, C

        # ====================================================
        # 1. 文本强化分支
        # ====================================================
        contexts_ = torch.cat([self.contexts2] * int(x[0].size()[0]), dim=0)
        text_embeddings = self.text_encoder(text.to(global_feat.device), contexts_)
        text_embeddings = text_embeddings.expand(B, -1, -1)
        
        # 如果你定义了 _enhance_text_features (ProText)，则调用它进行文本增强
        if hasattr(self, '_enhance_text_features'):
            text_embeddings = self._enhance_text_features(text_embeddings)
            
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # ====================================================
        # 2. 视觉分支：单时相 Mamba 扫描 (已加入残差保底机制)
        # ====================================================
        x_clip = []
        for i, feat in enumerate(x_orig):
            # 检查是否开启了 mamba 且存在单时相 mamba 模块
            if getattr(self, 'mamba_layers', False) and hasattr(self, 'mamba_modules_single') and i < len(self.mamba_modules_single):
                
                seq_feat = self.spatial_to_sequence(feat)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                #seq_out = self.mamba_modules_single[i](seq_feat) # 换个变量名 seq_out 接收输出
                h, w = feat.shape[2], feat.shape[3]
                seq_out = self.mamba_modules_single[i](seq_feat, H=h, W=w) 
                #feat_out = self.sequence_to_spatial(seq_out, h, w)
                feat_out = self.sequence_to_spatial(seq_out, h, w)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                # 加上残差连接！(原特征 + Mamba提取的增量特征)
                x_clip.append(feat + feat_out)
            else:
                # 兜底：如果没开 mamba 或越界，原样放入，保证计算图不断裂
                x_clip.append(feat)

        # ====================================================
        # 3. 计算 Score Map (100% 恢复原版：严格 L2 归一化)
        # ====================================================
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_norm)
        
        # 4. 将 score_map 拼接到对应的层级中
        x_clip[self.score_concat_index] = torch.cat([x_clip[self.score_concat_index], score_map], dim=1)

        # 此时返回的 x_clip 已经包含了：Mamba 提纯后的视觉特征 + 拼接好的 Score Map
        return text_embeddings, x_clip, score_map
    #2026-4-16-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
    def get_cls_text(self, img_infos, train=True):
        textA = []
        textB = []
        for i in range(len(img_infos)):
            try:
                foreA = ', '.join(['remote sensing image foreground objects']+img_infos[i].jsonA)
                foreB = ', '.join(['remote sensing image foreground objects']+img_infos[i].jsonB)
            except:
                foreA = ', '.join(['remote sensing image foreground objects']+img_infos[i]['jsonA'])
                foreB = ', '.join(['remote sensing image foreground objects']+img_infos[i]['jsonB'])
            backA = ', '.join(['remote sensing image background objects'])
            backB = ', '.join(['remote sensing image background objects'])

            textA.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA]]).unsqueeze(0))
            textB.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB]]).unsqueeze(0))
        return torch.cat(textA, dim=0), torch.cat(textB, dim=0)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        x_g = xA[-1][0]+xB[-1][0]
        x_l = xA[-1][1]+xB[-1][1]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])
        textA, textB = self.get_cls_text(data_samples)
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]

        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in range(len(x_clipA))]
        score_map_diff = score_mapA-score_mapB

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        
        # ====================================================
        # Mamba 融合扫描插入点
        # ====================================================
        if getattr(self, 'mamba_layers', False):
            
            x_fused = []
            for i, feat in enumerate(x):
                seq_feat = self.spatial_to_sequence(feat)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                #seq_out = self.mamba_modules_single[i](seq_feat) # 换个变量名 seq_out 接收输出
                h, w = feat.shape[2], feat.shape[3]
                seq_out = self.mamba_modules_fused[i](seq_feat, H=h, W=w)
                #feat_out = self.sequence_to_spatial(seq_out, h, w)
                feat_out = self.sequence_to_spatial(seq_out, h, w)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                
                # 加上残差连接！(原特征 + Mamba提纯的全局特征)
                x_fused.append(feat + feat_out) 
            x = x_fused
            
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        losses = dict()

        loss_decode = self._decode_head_forward_train_with_text(x, text_embeddingsA, text_embeddingsB, data_samples)
        losses.update(loss_decode)

        if self.with_identity_head:
            loss_identity_sm = self._identity_head_forward_train(
                score_map_diff/self.tau, data_samples, 'aux_score_map')
            losses.update(loss_identity_sm)
            loss_identity1 = self._identity_head_forward_train(
                x[0], data_samples, 'aux_layer0')
            losses.update(loss_identity1)
            loss_identity2 = self._identity_head_forward_train(
                x[1], data_samples, 'aux_layer1')
            losses.update(loss_identity2)
            loss_identity3 = self._identity_head_forward_train(
                x[2], data_samples, 'aux_layer2')
            losses.update(loss_identity3)
            loss_identity4 = self._identity_head_forward_train(
                x[3], data_samples, 'aux_layer3')
            losses.update(loss_identity4)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                _x_orig, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        import time
        torch.cuda.synchronize()
        start = time.time()
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        x_g = xA[-1][0]+xB[-1][0]
        x_l = xA[-1][1]+xB[-1][1]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])

        textA = []
        textB = []
        foreA = ', '.join(['remote sensing image foreground objects']+['mountain', 'bare land', 'ground track field', 'road', 'farmland', 'dense residential', 'island', 'highway', 'fertile land'])
        backA = ', '.join(['remote sensing image background objects'])
        foreB = ', '.join(['ground track field', 'farmland', 'bare land', 'wetland', 'golf course', 'island', 'fertile land', 'interchange', 'pond'])
        backB = ', '.join(['remote sensing image background objects'])
        textA.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA]]).unsqueeze(0))
        textB.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB]]).unsqueeze(0))
        textA, textB = torch.cat(textA, dim=0), torch.cat(textB, dim=0)

        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]

        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in range(len(x_clipA))]

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        
        # ====================================================
        # Mamba 融合扫描插入点 
        # ====================================================
        if getattr(self, 'mamba_layers', False):
            
            x_fused = []
            for i, feat in enumerate(x):
                seq_feat = self.spatial_to_sequence(feat)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                #seq_out = self.mamba_modules_single[i](seq_feat) # 换个变量名 seq_out 接收输出
                h, w = feat.shape[2], feat.shape[3]
                seq_out = self.mamba_modules_single[i](seq_feat, H=h, W=w) 
                #feat_out = self.sequence_to_spatial(seq_out, h, w)
                feat_out = self.sequence_to_spatial(seq_out, h, w)
                #2026-4-18-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
                
                # 加上残差连接！(原特征 + Mamba特征)
                x_fused.append(feat + feat_out) 
            x = x_fused
            
        x = [self.channel_att[i](x[i]) for i in range(len(x))]
        data_samples = [{'image_shape': (256, 256)}]

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, data_samples,
                                                      self.test_cfg)
        torch.cuda.synchronize()
        end = time.time()
        total_time = end - start
        print('total_time:{:.2f}'.format(total_time))
        return seg_logits

    def mm_slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:

        inputs = inputs[0].unsqueeze(0)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        imgA_pil = Image.open(batch_img_metas[0]['img_path'])
        imgB_pil = Image.open(batch_img_metas[0]['img_path'].replace('/A', '/B'))

        model, preprocess = init_clip()

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                cropA = imgA_pil.crop((x1, y1, x2, y2))
                cropB = imgB_pil.crop((x1, y1, x2, y2))
                jsonA, jsonB = clip_infer(cropA, cropB, model, preprocess)
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                batch_img_metas[0]['jsonA'] = jsonA
                batch_img_metas[0]['jsonB'] = jsonB
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole', 'mm_slide'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        elif self.test_cfg.mode == 'mm_slide':
            seg_logit = self.mm_slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred