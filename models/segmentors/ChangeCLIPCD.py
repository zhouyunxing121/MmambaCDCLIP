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

from mamba_ssm import Mamba

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=1):
    #def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba(
            d_model=dim,  # Model dimension
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def forward(self, x):
        """
        x: [B, N, C]  (sequence)
        returns: [B, N, C]
        """
        #return self.mamba(x)
        # Mamba前向
        out = self.mamba(x)
        #修改——————2026-2-25——————doubao——————ChangeCLIP模型定义
        # 增强数值稳定（防止nan）
        out = torch.clamp(out, min=-1e3, max=1e3)  # 缩小裁剪范围，避免过度裁剪
        out = torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=-1e3)  # 替换nan/inf
         #修改——————2026-2-25——————doubao——————ChangeCLIP模型定义
        return out
    #2026-2-18修改
    @property
    def d_model(self):
        """兼容属性访问：提供d_model别名(解决之前的属性访问错误)"""
        return self.dim
    #2026-2-18修改


@MODELS.register_module()
class ChangeCLIP(BaseSegmentor):
# noqa: E501
 

    def __init__(self,
                 backbone: ConfigType,
                 text_encoder: ConfigType,
                 context_decoder: ConfigType,
                 decode_head: ConfigType,
                 class_names=['remote sensing images', 'remote sensing images change area'],  #farmland change_area
                 context_length=5,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 tau=0.07,
                 identity_head=None,
                 token_embed_dim=512, 
                 #text_dim=1024,#RN50
                 text_dim=768,#VIT-L
                 #text_dim=512,
                
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 #增加Mamba 参数
                 mamba_layers=True, 
                 mamba_d_state=16, 
                 mamba_d_conv=4, 
                 mamba_expand=1,):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        """
        if pretrained is not None:
        
            #注意
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = '/home/dc001/.cache/clip/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained
            
            
            #text_encoder.pretrained = pretrained
            if backbone.get('pretrained') is not None:
                print('backbone.pretrained is :', backbone.pretrained)
            print('text_encoder.pretrained is :', text_encoder.pretrained)
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Cannot find pre-trained weight, using CLIP pre-trained weight")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            text_encoder.pretrained = '/home/dc001/.cache/clip/RN50.pt'

        """
        # === 新逻辑：明确分离 backbone 和 text_encoder 的预训练 ===
        if pretrained is not None:
        # pretrained 现在只用于 text_encoder（必须是 CLIP 权重）
        #   backbone 的预训练由 backbone 自己的 pretrained_mamba 控制
            if any(kw in pretrained for kw in ['RN50', 'RN101', 'ViT-B', 'clip','VIT-L']):
                text_encoder_pretrained = pretrained
            else:
            # 如果传入的不是 CLIP 权重，警告并使用默认 CLIP
                print(f"Warning: '{pretrained}' does not look like a CLIP checkpoint. "
                        f"Using default CLIP ViT-B-16 for text encoder.")
                text_encoder_pretrained = '/home/dc001/.cache/clip/ViT-B-16.pt'
                #text_encoder_pretrained = '/home/dc001/.cache/clip/ViT-B-16.pt'  # 512维
                # 或
                #text_encoder_pretrained = '/home/dc001/.cache/clip/ViT-L-14.pt'  # 768维
        else:
            print("pretrained is None. Using default CLIP for text encoder.")
            #text_encoder_pretrained = '/home/dc001/.cache/clip/RN50.pt'# 1024维
            text_encoder_pretrained = '/home/dc001/.cache/clip/ViT-L-14.pt'  # 768维
        #还要同步修改本文件、swin_text_head.py中的text_dim


        text_encoder.pretrained = text_encoder_pretrained
        print('text_encoder.pretrained is:', text_encoder_pretrained)

        # 注意：不再设置 backbone.pretrained！由 backbone 内部处理
        # （CLIPMambaWithAttention 已通过 pretrained_mamba 加载）
        self.backbone = MODELS.build(backbone)
        """
        #dummy_input = torch.zeros(1, 6, 224, 224)  # 单图 dummy input
        dummy_input = torch.zeros(1, 3, 224, 224)  
        """
        """
        # 根据 backbone 配置的 in_chans 动态创建 dummy input
        in_chans = backbone.get('in_chans', 3)
        dummy_input = torch.zeros(1, in_chans, 224, 224)
        with torch.no_grad():
            dummy_feats = self.backbone(dummy_input)
            # 获取前4个 stage 的通道数 (忽略最后的 [global, local])
            actual_feat_channels = [f.shape[1] for f in dummy_feats[:-1]]
        print("Detected backbone feature channels:", actual_feat_channels)
        """
        """
        if hasattr(backbone, 'dims'):
            actual_feat_channels = backbone['dims']  # [96, 192, 384, 768]
        else:
            #actual_feat_channels = [96, 192, 384, 768]  # 默认值
            actual_feat_channels = [96, 192, 384, 770]  
        print("Using fixed backbone feature channels:", actual_feat_channels)
        """
        actual_feat_channels = [96, 192, 384, 768]  
        print("Using fixed backbone feature channels:", actual_feat_channels)
        #actual_feat_channels2 = [96, 192, 384, 770]

       

        self.text_encoder = MODELS.build(text_encoder)
        self.context_decoder = MODELS.build(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index
      
        
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
        # self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        self.contexts2 = nn.Parameter(torch.randn(1, 1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts2)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        """
        self.minus_conv = nn.Sequential(
        ConvModule(
                    in_channels=self.minus_channel[0],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[1],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[2],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[3],
                    out_channels=256,
                    kernel_size=1)
                    )
        self.channel_att = nn.Sequential(SELayer(768, 256), SELayer(768, 256), SELayer(768, 256), SELayer(768, 256))
        """
        

        """
       
        actual_feat_channels2 = [96, 192, 384, 770]
        self.minus_conv = nn.ModuleList([  # 改为 ModuleList!
            ConvModule(
                in_channels=d, 
                out_channels=256, 
                kernel_size=1
                ) for d in actual_feat_channels
            ])

        self.channel_att = nn.ModuleList([  # 改为 ModuleList!
            #SELayer(d * 3 + 256, 256)  
            # x_orig*diff (d) + x_minus (256) + x_orig (d) → total = 2d + 256? 
            # 实际融合是: torch.cat([x_orig*x_diff (d), x_minus (256), x_orig (d)], dim=1) → d + 256 + d = 2d + 256
            SELayer(2 * d + 256, 256)
            for d in actual_feat_channels
            ])
        """

        #2026-1-11修改，见——看看原ChangeCLIP项目怎么解决的
        score_concat_index = 3
        score_channels = 256
        #2026-1-13修改——————见————————770通道特征图报错
        num_score_channels = self.num_classes  # = 2
        print("num_score_channels:", num_score_channels)
        #2026-1-13修改——————见————————770通道特征图报错
        """
        # x_clip 的通道数（backbone + 可选 score map）
        x_clip_channels = [
            ch + (score_channels if i == score_concat_index else 0)
            for i, ch in enumerate(actual_feat_channels)
        ]
        """

        #2026-1-13修改——————见————————770通道特征图报错
        x_clip_channels = [
            ch + (num_score_channels if i == self.score_concat_index else 0)
            for i, ch in enumerate(actual_feat_channels)
            ]
        print("x_clip_channels:", x_clip_channels)

        #2026-1-13修改——————见————————770通道特征图报错


        # minus_conv: 输入是 |xA - xB|，通道 = actual_feat_channels[i]
        self.minus_conv = nn.ModuleList([
            ConvModule(in_channels=ch, out_channels=score_channels, kernel_size=1)
            for ch in actual_feat_channels  
        ])

        # 融合后通道: [x*x_diff (C), x_minus (256), x (C)] → 2*C + 256, 其中 C = x_clip_channels[i]
        #fused_channels = [2 * ch + score_channels for ch in x_clip_channels]
        fused_channels = [4 * ch + score_channels for ch in x_clip_channels]
        print("fused_channels:", fused_channels)

        self.channel_att = nn.ModuleList([
            #SELayer(2*ch+256, 256) for ch in fused_channels 
            #SELayer(ch, 256) for ch in fused_channels
            #ratio=16
            #out_channels=
            SELayer(ch, out_channels=256,ratio=16) for ch in fused_channels
            #SELayer(ch, out_channels=None,ratio=16) for ch in fused_channels
            #2026-1-29修改——————见————————SELayer_v2使用有误
            #SELayer(ch, reduction=256) for ch in fused_channels

        ])
        #2026-1-11修改，见——看看原ChangeCLIP项目怎么解决的

        """
        self.mamba_layers = mamba_layers
        if mamba_layers:
        # 为每个空间层级（如 4 个尺度）创建 Mamba 层
            mamba_dims = [256, 512, 1024, 2048]  
            self.mamba_modules = nn.ModuleList([
                MambaLayer(dim=d, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                #MambaLayer(dim=768, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                #x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1)],
                for d in mamba_dims
            ])
        """
    
        self.mamba_layers = mamba_layers
        """
        if mamba_layers:
            #mamba_dims = [256, 512, 1024, 2048]  
            #与 backbone 输出 [96,192,384,768] 不匹配

            # 第一组：用于单时相特征（xA 和 xB 分别处理）
            self.mamba_modules_single = nn.ModuleList([
                MambaLayer(dim=d, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                for d in mamba_dims
            ])

            # 第二组：用于融合后的特征（x_orig * weight + minus 特征拼接后）
            self.mamba_modules_fused = nn.ModuleList([
                MambaLayer(dim=d * 3, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                #for d in mamba_dims  
                for d in [256, 512, 1024, 2048]
            ])
        """

        # 用实际通道数初始化 Mamba 模块
        if self.mamba_layers:
            # 第一组：单时相 Mamba (通道数 = actual_feat_channels)
            self.mamba_modules_single = nn.ModuleList([
                MambaLayer(dim=d, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                for d in actual_feat_channels
            ])
            # 第二组：融合后 Mamba (通道数 = 3 * actual_feat_channels, 因拼接了3部分)
            """
            self.mamba_modules_fused = nn.ModuleList([
                MambaLayer(dim=2*d + 256, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                #MambaLayer(dim=d * 3, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                for d in actual_feat_channels
            ])
            """
            ##2026-1-11修改，见——看看原ChangeCLIP项目怎么解决的
            # Mamba fused 模块也需用 fused_channels
            ## 融合后通道: [x*x_diff (C), x_minus (256), x (C)] → 2*C + 256, 其中 C = x_clip_channels[i]
           

            #self.mamba_modules_fused = nn.ModuleList([
             #MambaLayer(dim=ch,  d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) 
             #for ch in fused_channels
            #])
            #dim=2*ch + 256
        #2026-1-11修改，见—————看看原ChangeCLIP项目怎么解决的

        #修改——————2026-2-25---doubao——————ChangeCLIP模型定义
            # 第二组：融合后 Mamba (通道数 = SELayer 输出的 256)
            self.mamba_modules_fused = nn.ModuleList([
                MambaLayer(dim=256,  d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) 
                for _ in fused_channels  # 只需要保持数量一致，维度固定为256
            ])
        #2026-1-11修改，见—————doubao———看看原ChangeCLIP项目怎么解决的

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        #从图像中提取特征
        x = self.backbone(inputs)
        return x
    


    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        #空间融合
        #将输入的6通道图像（3通道A + 3通道B）拆分为两个3通道图像
        #前3通道为时相A
        #后3通道为时相B
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        #特征提取
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)
        #使用 CLIP 模型对每个时相的特征进行文本引导的增强，得到更语义化的视觉特征 x_clipA 和 x_clipB
        x_g = xA[-1][0]+xB[-1][0]
        x_l = xA[-1][1]+xB[-1][1]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])
        textA, textB = self.get_cls_text(batch_img_metas, False)

        """
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)
        #多尺度空间融合
        # (1) 原始特征拼接
        #将A和B在每个尺度上的特征图在通道维度上拼接。直接空间融合
        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]
        # (2) 差异特征提取
        #计算A和B特征图的绝对差值，并通过1x1卷积降维
        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        #索引访问 [i]	应改为 nn.ModuleList，否则会报错！修正建议：self.minus_conv = nn.ModuleList([...])
        #计算A和B特征图的余弦相似度，并转换为差异权重图
        x_diff = [
            F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) 
            for i in range(len(x_clipA))
            ]
        """
        #2026-1-13修改——————见————————770通道特征图报错
        text_embeddingsA, x_clipA_raw, x_clipA_fused, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB_raw, x_clipB_fused, score_mapB = self.after_extract_feat_clip(xB, textB)

        # 用 fused 版本做拼接
        x_orig = [
            torch.cat([x_clipA_fused[i], x_clipB_fused[i]], dim=1) 
            for i in range(len(x_clipA_fused))
            ]

        # 用 raw 版本做差异
        x_minus = [
            self.minus_conv[i](torch.abs(x_clipA_raw[i] - x_clipB_raw[i])) 
            for i in range(len(x_clipA_raw))
            ]
        x_diff = [
            F.sigmoid(1 - torch.cosine_similarity(x_clipA_raw[i], x_clipB_raw[i], dim=1)).unsqueeze(1) 
            for i in range(len(x_clipA_raw))
            ]
        
        #2026-1-13修改——————见————————770通道特征图报错


        score_map_diff = score_mapA-score_mapB
        
        losses = dict()
        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig
        #融合策略
        #x 初始为 x_orig（拼接特征）
        #x[i] * x_diff[i]：原始拼接特征乘以差异权重。
        #x_minus[i]：显式提取的差异特征。
        #x[i]：原始拼接特征。
        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        #通道注意力加权
        x = [self.channel_att[i](x[i]) for i in range(len(x))]
        #索引访问 [i]	应改为 nn.ModuleList，否则会报错！修正建议：self.channel_att = nn.ModuleList([...])
        #loss()也用到了x_orig,x_minus,x_diff这些
        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig
            #x_for_neck = xA[0:4]  # 或 xB[0:4]
            #x_orig_neck = list(self.neck(x_for_neck))
            #_x_orig = x_orig_neck

        # 在 跨时相融合后 再使用 Mamba 建模全局变化上下文。也可以考虑在loss()添加。
        if self.mamba_layers:
            x_fused = []
            for i, feat in enumerate(x):
                seq = self.spatial_to_sequence(feat)
                #print(f"[DEBUG] Stage {i}: input seq shape = {seq.shape}, "f"expected d_model = {self.mamba_modules_fused[i].d_model}")
                seq = self.mamba_modules_fused[i](seq)
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq, h, w)
                x_fused.append(feat)
            x = x_fused
            
        
        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, batch_img_metas,
                                              self.test_cfg)

        return seg_logits


    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_train_with_text(self, x, textA, textB: List[Tensor],
                                            data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss_changeclip(x, textA, textB, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
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
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.loss(
            x, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_aux, loss_id))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit
    def spatial_to_sequence(self, x):
        """
        x: [B, C, H, W]
        returns: [B, N, C], where N = H * W
        """
        B, C, H, W = x.shape
        x = x.flatten(2)  # -> [B, C, H*W]
        x = x.transpose(1, 2)  # -> [B, H*W, C]
        return x

    def sequence_to_spatial(self, x, H, W):
        """
        x: [B, N, C]
        returns: [B, C, H, W]
        """
        B, N, C = x.shape
        x = x.transpose(1, 2)  # -> [B, C, N]
        x = x.reshape(B, C, H, W)
        return x
    def after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
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

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map
    """
    def after_extract_feat_clip(self, x, text):
        # x 是 backbone 输出的多尺度特征列表: [feat1, feat2, feat3, feat4, (global, local)]
        x_orig = list(x[0:4])## 前4个尺度的特征图 [B,C,H,W]
        global_feat, visual_embeddings = x[4]## CLIP 的全局+局部特征

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([
                global_feat.reshape(B, C, 1), 
                visual_embeddings.reshape(B, C, H*W)
                ], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        contexts_ = torch.cat([self.contexts2] * int(x[0].size()[0]), dim=0)
        text_embeddings = self.text_encoder(text.to(global_feat.device), contexts_).expand(B, -1, -1)
        # text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        

        # update text_embeddings by visual_context!更新文本嵌入
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        #对每个尺度特征应用 Mamba
        x_clip = []
        for i, feat in enumerate(x_orig):
            if self.mamba_layers and i < len(self.mamba_modules_single):
            # 转为序列
                seq_feat = self.spatial_to_sequence(feat)  # [B, H*W, C]
            # 应用 Mamba
                seq_feat = self.mamba_modules_single[i](seq_feat)  # [B, H*W, C]
            # 转回空间形式
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq_feat, h, w)  # [B, C, H, W]
            x_clip.append(feat)
        
        # compute score map and concat  #计算 score map 并拼接
        #B, K, C = text_embeddings.shape
        #visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        #text = F.normalize(text_embeddings, dim=2, p=2)
        #score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        #x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        
        # 计算 score map 并拼接
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_norm)

        # 将 score_map 拼接到指定层
        x_clip[self.score_concat_index] = torch.cat([x_clip[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_clip, score_map
    """
    #2026-1-13修改——————见————————770通道特征图报错
    def after_extract_feat_clip(self, x, text):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([
                global_feat.reshape(B, C, 1), 
                visual_embeddings.reshape(B, C, H*W)
            ], dim=2).permute(0, 2, 1)

        contexts_ = torch.cat([self.contexts2] * int(x[0].size()[0]), dim=0)
        text_embeddings = self.text_encoder(text.to(global_feat.device), contexts_).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # === 构建原始特征（用于差异计算）===
        x_clip_raw = []
        for i, feat in enumerate(x_orig):
            if self.mamba_layers and i < len(self.mamba_modules_single):
                seq_feat = self.spatial_to_sequence(feat)
                seq_feat = self.mamba_modules_single[i](seq_feat)
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq_feat, h, w)
            x_clip_raw.append(feat)

        # === 计算 score_map ===
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_norm)  # [B, 2, H, W]

        # === 构建融合特征（用于拼接）===
        x_clip_fused = [f.clone() for f in x_clip_raw]  # copy
        x_clip_fused[self.score_concat_index] = torch.cat([
            x_clip_fused[self.score_concat_index], score_map
        ], dim=1)

        return text_embeddings, x_clip_raw, x_clip_fused, score_map
    #2026-1-13修改——————见————————770通道特征图报错

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
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
    
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        x_g = xA[-1][0]+xB[-1][0]
        x_l = xA[-1][1]+xB[-1][1]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])
        textA, textB = self.get_cls_text(data_samples)

        """
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)
        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]
        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in range(len(x_clipA))]
        """
        #2026-1-13修改——————见————————770通道特征图报错
        text_embeddingsA, x_clipA_raw, x_clipA_fused, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB_raw, x_clipB_fused, score_mapB = self.after_extract_feat_clip(xB, textB)
        # 用 fused 版本做拼接
        x_orig = [torch.cat([x_clipA_fused[i], x_clipB_fused[i]], dim=1) for i in range(len(x_clipA_fused))]
        # 用 raw 版本做差异
        x_minus = [self.minus_conv[i](torch.abs(x_clipA_raw[i] - x_clipB_raw[i])) for i in range(len(x_clipA_raw))]
        x_diff = [
            F.sigmoid(1 - torch.cosine_similarity(x_clipA_raw[i], x_clipB_raw[i], dim=1)).unsqueeze(1) 
            for i in range(len(x_clipA_raw))
            ]
        
        
        #2026-1-13修改——————见————————770通道特征图报错


        score_map_diff = score_mapA-score_mapB

        

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        losses = dict()

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig
            #x_for_neck = xA[0:4]  # 或 xB[0:4]，两者对称
            #x_orig_neck = list(self.neck(x_for_neck))
            #_x_orig = x_orig_neck

        loss_decode = self._decode_head_forward_train_with_text(x, text_embeddingsA, text_embeddingsB, data_samples)
        losses.update(loss_decode)

        if self.with_identity_head:
            loss_identity_sm = self._identity_head_forward_train(
                score_map_diff/self.tau, data_samples, 'aux_score_map')
            losses.update(loss_identity_sm)
            loss_identity1 = self._identity_head_forward_train(
                x[0], data_samples, 'aux_layer0')
            losses.update(loss_identity1)
            # loss_identity1 = self._identity_head_forward_train(
            #     x[0], data_samples, 'aux_layer0')
            # losses.update(loss_identity1)
            loss_identity2 = self._identity_head_forward_train(
                x[1], data_samples, 'aux_layer1')
            losses.update(loss_identity2)
            loss_identity3 = self._identity_head_forward_train(
                x[2], data_samples, 'aux_layer2')
            losses.update(loss_identity3)
            loss_identity4 = self._identity_head_forward_train(
                x[3], data_samples, 'aux_layer3')
            losses.update(loss_identity4)
            # loss_identity4 = self._identity_head_forward_train(
            #     x[3], data_samples, 'aux_layer3')
            # losses.update(loss_identity4)

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
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
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
        """
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        x_orig = [
        torch.cat([x_clipA[i], x_clipB[i]], dim=1) 
        for i in range(len(x_clipA))
        ]

        x_minus = [
        self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) 
        for i in range(len(x_clipA))
        ]
        x_diff = [
        F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) 
        for i in range(len(x_clipA))
        ]
        """


        #2026-1-13修改——————见————————770通道特征图报错
        text_embeddingsA, x_clipA_raw, x_clipA_fused, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB_raw, x_clipB_fused, score_mapB = self.after_extract_feat_clip(xB, textB)

        # 注意：后续使用 x_clipA_fused / x_clipB_fused 进行拼接
        x_orig = [
            torch.cat([x_clipA_fused[i], x_clipB_fused[i]], dim=1) 
            for i in range(len(x_clipA_fused))
            ]
        x_minus = [
            self.minus_conv[i](torch.abs(x_clipA_raw[i] - x_clipB_raw[i])) 
            for i in range(len(x_clipA_raw))
            ]
        x_diff = [
            F.sigmoid(1 - torch.cosine_similarity(x_clipA_raw[i], x_clipB_raw[i], dim=1)).unsqueeze(1) 
            for i in range(len(x_clipA_raw))
            ]





        #2026-1-13修改——————见————————770通道特征图报错

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig
            #x_for_neck = xA[0:4]  # 或 xB[0:4]
            #x_orig_neck = list(self.neck(x_for_neck))
            #_x_orig = x_orig_neck

        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
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