custom_imports = dict(
    imports=[
        'mmseg.models.backbones.clip_backbone',
        'mmseg.models.segmentors.ChangeCLIPCD'
        ],
    allow_failed_imports=False
)
_base_ = [
    '../_base_/datasets/base_cd.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py']
import os
data_root = '/online1/wangshiying_group/wangshiying/gjx/data/SYSU-CD'
#data_root = '/home/dc001/data/SYSU-CD'
metainfo = dict(
                #classes=('unchange', 'change'),
                #classes=('background', 'urban construction and land cover change')
                classes=('background', 'building'),
                palette=[[0, 0, 0], [255, 255, 255]])

crop_size = (256, 256)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',#mmseg\models\data_preprocessor.py
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])


#2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
norm_cfg = dict(type='SyncBN', requires_grad=True)
#norm_cfg = dict(type='BN', requires_grad=True)
#2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析

#2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
find_unused_parameters = True
#find_unused_parameters = False
#2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
model = dict(
    type='ChangeCLIP',
    #pretrained=None,
    #pretrained='/home/dc001/.cache/clip/RN50.pt',
    pretrained='/online1/wangshiying_group/wangshiying/.cache/clip/RN50.pt',
    context_length=64,
    text_head=False,
    
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        #output_dim=768,
        input_resolution=512,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',#clip_backbone
        context_length=77,
        #embed_dim=768,#VIT-L
       
        embed_dim=1024,#RN50
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',#clip_backbone
        context_length=16,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        #visual_dim=768,
        
        visual_dim=1024,
        dropout=0.1,
        #outdim=768,#VIT-L
       
        outdim=1024,#RN50
        #outdim=4, # 改为 4 以匹配 Neck 的 1540 (768*2 + 4)
        # ————2026-3-21——————修改——————见——————ChangeCLIP模型配置（SYSU-CD数据集）
        style='pytorch'),
    #特征融合或特征提取
    neck=dict(
        type='FPN',#mmseg\models\necks\fpn.py

        in_channels=[512, 1024, 2048, 4100],
        #in_channels=[192, 384, 768, 1540],  # 每个都 *2，最后一层 +2 再 *2
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='SwinTextDecode',
        in_channels=[256, 256, 256, 256],#与neck.out_channels保持一致
        in_index=[0, 1, 2, 3],#与neck.out_channels保持一致
        feature_strides=[4, 8, 16, 32],
        #text_dim=768,
        text_dim=1024,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_classes=2,
        channels=256,
        #2026-4-19-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
        #loss_decode=dict(
        #    type='CrossEntropyLoss',
        #    use_sigmoid=False,
        #    loss_weight=1.0,
            
        #    #avg_non_ignore=True,#修改2026-3-2——————遥感图像变化检测配置解析
        #)
        # 核心修改：交叉熵 (聚焦像素分类) + Dice (聚焦目标面积，解决不平衡)
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_name='loss_ce',
                loss_weight=1.0,
                avg_non_ignore=True), 
            
            #dict(
            #    type='DiceLoss',
            #    loss_name='loss_dice',
            #    loss_weight=1.0,
            #    use_sigmoid=False, # 多类别保持 False
            #    class_weight=[0.0, 1.0] #告诉模型忽略背景(0)，只算前景建筑(1)的Dice！
            #)
            ## 换成更稳健且同样优化 IoU 的 LovaszLoss！
            dict(
                type='LovaszLoss',
                loss_name='loss_lovasz',
                loss_weight=1.0,
                reduction='none') # reduction='none' 确保它能和 CE 平滑相加       
        ]
        #2026-4-19-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
    ),

    # === 启用 Mamba 增强模块 ===
    mamba_layers=True,
    #mamba_layers=False,
    #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
    mamba_d_state=16,
    
    #2026-4-8-修改——————见————————Gemini——————ChangeCLIP 项目介绍与解析
    mamba_d_conv=4,
    mamba_expand=2,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    #type='OptimWrapper',
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00003, 
        #lr=0.000006, 
        #lr=0.000003, 
        betas=(0.9, 0.999), 
        weight_decay=0.01),
       
        #修改2026-3-2——————遥感图像变化检测配置解析
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            # ————2026-3-21——————修改——————见——————ChangeCLIP模型配置（SYSU-CD数据集）
            #'backbone': dict(lr_mult=0.1),       # 预训练主干降低学习率
            #'text_encoder': dict(lr_mult=0.1),   # 预训练文本编码器降低学习率
            #'decode_head': dict(lr_mult=10.0),   # 解码头随机初始化，提高学习率加速收敛
            #'neck': dict(lr_mult=10.0),          # Neck 随机初始化，提高学习率
            #'context_decoder': dict(lr_mult=10.0),
            # ————2026-3-21——————修改——————见——————ChangeCLIP模型配置（SYSU-CD数据集）
        }
    ),
    #修改2026-3-2——————遥感图像变化检测配置解析
    #clip_grad=dict(max_norm=1.0, norm_type=2)  # 添加梯度裁剪
    #clip_grad=dict(max_norm=0.01, norm_type=2)
)

param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=False, 
        begin=0, 
        end=1500#预热步数为总步数的 5% 左右
        ),
        #修改2026-3-2——————遥感图像变化检测配置解析
    dict(
        type='PolyLR',
        #修改2026-3-2——————遥感图像变化检测配置解析
        eta_min=0.0,
        #eta_min=1e-7,
        power=1.0,
        #power=0.9,
        #修改2026-3-2——————遥感图像变化检测配置解析
        begin=1500,
        #预热步数为总步数的 5% 左右
        end=20000,
        by_epoch=False,
    )
]

train_dataloader = dict(
    batch_size=20,
    #batch_size=4,
    #batch_size=8,
    num_workers=8,
    #num_workers=4,
    #num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.txt'))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    #num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.txt'))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    #num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.txt'))

# training schedule for 120k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')) 