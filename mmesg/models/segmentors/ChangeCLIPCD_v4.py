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
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv,  
            expand=expand,  
        )

    def forward(self, x):
        """
        x: [B, N, C]  (sequence)
        returns: [B, N, C]
        """
        
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
                 #context_length=64,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 protext_feat_path=None,
                 tau=0.07,
                 identity_head=None,
                 token_embed_dim=512, 
                 text_dim=1024,#RN50
                 #text_dim=768,#VIT-L
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
                 mamba_expand=1,
                 ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
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
        #2026-4-10-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        #actual_feat_channels = [96, 192, 384, 768]  
        #print("Using fixed backbone feature channels:", actual_feat_channels)
        #actual_feat_channels2 = [96, 192, 384, 770]
        device = next(self.backbone.parameters()).device # 获取设备
        # 创建一个 3通道的 Dummy 图像，尺寸随意，比如 256x256
        dummy_input = torch.zeros(2, 3, 256, 256).to(device) 
        
        with torch.no_grad():
            dummy_feats = self.backbone(dummy_input)
            
        # dummy_feats 是一个 list: [feat1, feat2, feat3, feat4, (global, local)]
        # 我们只取前 4 个空间尺度的特征，提取它们的通道数 (索引为 1 的维度)
        actual_feat_channels = [feat.shape[1] for feat in dummy_feats[:4]]
        
        print(f"Dynamically detected backbone feature channels: {actual_feat_channels}")
       
        #2026-4-10-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
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
        # 2026-4-12====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析=========
        #self.contexts2 = nn.Parameter(torch.randn(1, 1, context_length, token_embed_dim))
        #nn.init.trunc_normal_(self.contexts2)
        #self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        # 1. 动态获取正确的文本维度 (从 text_encoder 配置中取，优先使用传进来的 text_dim)
        # 2026-4-14====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析=========
        actual_text_dim = text_encoder.get('embed_dim', text_dim)

        self.contexts2 = nn.Parameter(torch.randn(1, 1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts2)
        
        # 2. 使用动态获取的维度初始化 gamma
        self.gamma = nn.Parameter(torch.ones(actual_text_dim) * 1e-4)
        # 2026-4-14====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析=========
        print(f"Dynamically set text dimension to: {actual_text_dim}")
        # 2026-4-12====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析=========
        # 2026-4-8====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析=========
        # 引入 ProText 辅助增强模块

        # 2026-4-10====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析======
        self.use_protext = False  # 默认不使用 ProText

        if protext_feat_path is not None:
            if os.path.exists(protext_feat_path):
                try:
                    expert_feats = torch.load(protext_feat_path) # [2, D]
                    # 注册为 Buffer，这样 to(cuda) 时会自动移动，且不参与求导
                    self.register_buffer('protext_embeddings', expert_feats)
                    
                    # 定义可学习的融合参数 (初始值为 0.0，Sigmoid 后为 0.5)
                    self.protext_alpha_raw = nn.Parameter(torch.tensor(0.0))
                    
                    self.use_protext = True # 只有成功加载且无报错，才设为 True
                    print(f" [ProText] Successfully loaded expert features from {protext_feat_path}")
                    print(f" [ProText] Feature shape: {expert_feats.shape}")
                except Exception as e:
                    print(f" [ProText] Failed to load features from {protext_feat_path}. Error: {e}")
                    print(" [ProText] Falling back to original CLIP text features.")
            else:
                print(f" [ProText] File not found: {protext_feat_path}")
                print("[ProText] Falling back to original CLIP text features.")
        else:
            # 配置文件中没有提供 protext_feat_path，正常静默跳过，使用纯 CLIP
            pass

        # 2026-4-10====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析======
        # 2026-4-8====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析=========
        

        #2026-1-11修改，见——看看原ChangeCLIP项目怎么解决的
        score_concat_index = 3
        score_channels = 256
        #2026-1-13修改——————见————————770通道特征图报错
        num_score_channels = self.num_classes  # = 2
        print("num_score_channels:", num_score_channels)
        #2026-1-13修改——————见————————770通道特征图报错
        


        #2026-1-13修改——————见————————770通道特征图报错
        x_clip_channels = [
            ch + (num_score_channels if i == self.score_concat_index else 0)
            for i, ch in enumerate(actual_feat_channels)
            ]
        print(f" Channels after score_map concat (x_clip_channels): {x_clip_channels}")
        #2026-1-13修改——————见————————770通道特征图报错


        # minus_conv: 输入是 |xA - xB|，通道 = actual_feat_channels[i]
        self.minus_conv = nn.ModuleList([
            ConvModule(in_channels=ch, out_channels=score_channels, kernel_size=1)
            #for ch in actual_feat_channels 
            for ch in x_clip_channels 
        ])


        fused_dim=768

        self.channel_att = nn.ModuleList([
            SELayer(fused_dim, out_channels=256, ratio=16) for _ in range(4)
        ])
        #2026-1-11修改，见——看看原ChangeCLIP项目怎么解决的

        self.mamba_layers = mamba_layers
        
        # 用实际通道数初始化 Mamba 模块
        if self.mamba_layers:
            # 第一组：单时相 Mamba (通道数 = actual_feat_channels)
            self.mamba_modules_single = nn.ModuleList([
                MambaLayer(dim=d, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                for d in actual_feat_channels
            ])
            # 第二组：融合后 Mamba (通道数 = 3 * actual_feat_channels, 因拼接了3部分)

            ##2026-1-11修改，见——看看原ChangeCLIP项目怎么解决的

        #2026-1-11修改，见—————看看原ChangeCLIP项目怎么解决的

        #修改——————2026-2-25---doubao——————ChangeCLIP模型定义
            # 第二组：融合后 Mamba (通道数 = SELayer 输出的 256)
            '''
            self.mamba_modules_fused = nn.ModuleList([
                MambaLayer(dim=256,  d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) 
                for _ in fused_channels  # 只需要保持数量一致，维度固定为256
            ])
            '''
            #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
            # 新修改：让融合后的 Mamba 在被 SE Layer 降维之前，以高维通道进行全局特征建模
            #self.mamba_modules_fused = nn.ModuleList([
            #    MambaLayer(dim=ch, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) 
            #    for ch in fused_channels  # 之前是写死的 256，现在改为 fused_channels
            #])
            self.mamba_modules_fused = nn.ModuleList([
                MambaLayer(dim=fused_dim, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand) 
                for _ in range(4)
            ])

            #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
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
    
    #2026-4-12-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
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

        # 1. 提取包含 score_map 的时相特征
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        # 2. 计算差值特征 (256维) 和 相似度特征 (1维)
        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in range(len(x_clipA))]
        
        # 3. 高维超级特征拼接 (比如 1540 维)
        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]

        # 4. FPN 降维 (1540 -> 256)
        if getattr(self, 'with_neck', False):
            x_neck = list(self.neck(x_orig))
            _x_orig = x_neck 
            x_orig = x_neck

        # 5. 组装最终特征 (x)
        if getattr(self, 'text_head', False):
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        # 6. 将 256 维的特征与 diff 和 minus 拼接，构成 768 维的特征空间
        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        
        # 7. 融合后 Mamba 全局扫描 (在 768 维空间)
        if getattr(self, 'mamba_layers', False):
            x_fused = []
            for i, feat in enumerate(x):
                seq = self.spatial_to_sequence(feat)
                seq = self.mamba_modules_fused[i](seq)
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq, h, w)
                x_fused.append(feat)
            x = x_fused

        # 8. 通道注意力加权并降维 (768 -> 256)
        x = [self.channel_att[i](x[i]) for i in range(len(x))]
        
        # 9. 送入解码器
        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, batch_img_metas, self.test_cfg)

        return seg_logits
    #2026-4-12-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
    
    
    
    
    def get_enhanced_text_embeddings(self, classnames):
        """
        获取并融合基础 CLIP 文本特征与 ProText 专家特征
        """
        # 1. 基础特征：让基础 CLIP Text Encoder 生成 (携带通用常识)
        # base_text_features 形状应为 [2, 768]
        base_text_features = self.text_encoder(classnames) 
        
        # 必须归一化
        base_text_features = F.normalize(base_text_features, dim=-1)

        if self.use_protext:
            # 2. 获取当前的融合权重 alpha (0 到 1 之间)
            alpha = torch.sigmoid(self.protext_alpha_raw)
            
            # 3. 专家特征
            expert_text_features = self.protext_embeddings
            
            # 4. === 核心：球面上或线性特征插值 ===
            # 将通用特征与专家特征在特征空间进行融合
            enhanced_text_features = (1.0 - alpha) * base_text_features + alpha * expert_text_features
            
            # 5. 融合后必须再次 L2 归一化，维持余弦相似度的数学性质
            enhanced_text_features = F.normalize(enhanced_text_features, dim=-1)
            
            return enhanced_text_features
        else:
            return base_text_features
    

    def _enhance_text_features(self, base_text_embeddings):
        """
        核心融合逻辑：将基础 CLIP 特征与 ProText 专家特征在单位超球面上插值
        base_text_embeddings: [B, 2, C]
        """
        if not getattr(self, 'use_protext', False) or not hasattr(self, 'protext_embeddings'):
            return base_text_embeddings

        # 1. 基础特征 L2 归一化
        base_norm = F.normalize(base_text_embeddings, dim=-1)
        
        # 2. 计算当前的融合比例 (0 ~ 1)
        alpha = torch.sigmoid(self.protext_alpha_raw)
        
        # 3. 专家特征 L2 归一化
        expert_norm = F.normalize(self.protext_embeddings, dim=-1) # [2, C]
        
        # 4. 维度对齐: expert_norm 从 [2, C] 扩展到 [B, 2, C]
        if base_norm.dim() == 3 and expert_norm.dim() == 2:
            expert_norm = expert_norm.unsqueeze(0).expand(base_norm.shape[0], -1, -1)
            
        # 兜底检查维度
        assert base_norm.shape[-1] == expert_norm.shape[-1], \
            f"维度不匹配! Base: {base_norm.shape[-1]}, Expert: {expert_norm.shape[-1]}"
        
        # 5. 球面插值融合
        enhanced_text_embeddings = (1.0 - alpha) * base_norm + alpha * expert_norm
        
        # 6. 融合后必须再次 L2 归一化
        return F.normalize(enhanced_text_embeddings, dim=-1)

# 2026-4-8====修改=========见=======Gemini=============ChangeCLIP 项目介绍与解析=========
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
        #text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
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

        #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析

        # 1. 文本强化：获取原版文本特征，并用 ProText 进行球面插值增强
        contexts_ = torch.cat([self.contexts2] * int(x[0].size()[0]), dim=0)
        text_embeddings = self.text_encoder(text.to(global_feat.device), contexts_).expand(B, -1, -1)
        text_embeddings = self._enhance_text_features(text_embeddings)
        #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # === 构建原始特征（用于差异计算）===
        # 2. 单时相 Mamba 扫描：在纯视觉特征上寻找规律
        x_clip = []
        for i, feat in enumerate(x_orig):
            if getattr(self, 'mamba_layers', False) and i < len(self.mamba_modules_single):
                seq_feat = self.spatial_to_sequence(feat)
                seq_feat = self.mamba_modules_single[i](seq_feat)
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq_feat, h, w)
            x_clip.append(feat)

        # === 计算 score_map ===
        #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        logit_scale = getattr(self.text_encoder, 'logit_scale', None)
        scale = logit_scale.exp() if logit_scale is not None else 100.0
        
        # 计算带温度缩放的相似度
        #score_map = scale * torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_norm)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_norm)
        #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        # === 构建融合特征（用于拼接）===
        # 直接将 score_map 拼接到特征里
        x_clip[self.score_concat_index] = torch.cat([x_clip[self.score_concat_index], score_map], dim=1)

        # 此时的 x_clip 通道数为 [96, 192, 384, 770]
        return text_embeddings, x_clip, score_map
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

       
        #2026-1-13修改——————见————————770通道特征图报错
        #2026-4-9-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        # 1. 提取包含 score_map 的时相特征
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        # 2. 基础拼接 (用于送入 FPN) -> 通道数为 [192, 384, 768, 1540]
        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]

        # 3. 计算差值特征 (256维) 和 相似度特征 (1维)
        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in range(len(x_clipA))]
        #2026-4-9-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        #2026-1-13修改——————见————————770通道特征图报错


        score_map_diff = score_mapA-score_mapB
        # 4. FPN 降维 (1540 -> 256)
        if getattr(self, 'with_neck', False):
            #x_orig = list(self.neck(x_orig))
            #_x_orig = x_orig
            x_neck = list(self.neck(x_orig))
            _x_orig = x_neck # 用于 aux_head
            x_orig = x_neck

        losses = dict()
        if getattr(self, 'text_head', False):
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig
        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        #2026-4-9-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
      
        if getattr(self, 'mamba_layers', False):
            x_fused = []
            for i, feat in enumerate(x):
                seq = self.spatial_to_sequence(feat)
                seq = self.mamba_modules_fused[i](seq)
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq, h, w)
                x_fused.append(feat)
            x = x_fused

        #2026-4-9-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        losses = dict()

        #if self.with_neck:
        #    x_orig = list(self.neck(x_orig))
        #    _x_orig = x_orig
            #x_for_neck = xA[0:4]  # 或 xB[0:4]，两者对称
            #x_orig_neck = list(self.neck(x_for_neck))
            #_x_orig = x_orig_neck
#2026-4-6-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        '''
        if self.mamba_layers:
            x_fused = []
            for i, feat in enumerate(x):
                seq = self.spatial_to_sequence(feat)
                seq = self.mamba_modules_fused[i](seq)
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq, h, w)
                x_fused.append(feat)
            x = x_fused
        '''
#2026-4-6-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析

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
#2026-4-12-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
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

        # 1. 提取包含 score_map 的时相特征
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(xB, textB)

        # 2. 计算差值特征 (256维) 和 相似度特征 (1维)
        x_minus = [self.minus_conv[i](torch.abs(x_clipA[i]-x_clipB[i])) for i in range(len(x_clipA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)).unsqueeze(1) for i in range(len(x_clipA))]

        # 3. 基础拼接 (用于送入 FPN) -> 通道数为 [192, 384, 768, 1540] 等
        x_orig = [torch.cat([x_clipA[i], x_clipB[i]], dim=1) for i in range(len(x_clipA))]

        # 4. FPN 降维 (降至 256)
        if getattr(self, 'with_neck', False):
            x_neck = list(self.neck(x_orig))
            _x_orig = x_neck
            x_orig = x_neck

        # 5. 组装特征
        if getattr(self, 'text_head', False):
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        # 6. 再次拼接构建 768 维超级特征
        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        
        # 7. 融合后 Mamba 全局扫描
        if getattr(self, 'mamba_layers', False):
            x_fused = []
            for i, feat in enumerate(x):
                seq = self.spatial_to_sequence(feat)
                seq = self.mamba_modules_fused[i](seq)
                h, w = feat.shape[2], feat.shape[3]
                feat = self.sequence_to_spatial(seq, h, w)
                x_fused.append(feat)
            x = x_fused
            
        # 8. 通道注意力加权 (768 -> 256)
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        # 9. 动态生成 Dummy data_samples 兜底
        if data_samples is None:
            data_samples = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0]
                )
            ] * inputs.shape[0]

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, data_samples, self.test_cfg)
        
        torch.cuda.synchronize()
        end = time.time()
        print('total_time:{:.2f}'.format(end - start))
        return seg_logits
#2026-4-12-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
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