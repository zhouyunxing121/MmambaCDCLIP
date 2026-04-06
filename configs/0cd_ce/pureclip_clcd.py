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
    '../_base_/schedules/schedule_160k.py'
]

import os
data_root = '/home/dc001/gjx/rschange/data/CLCD'

metainfo = dict(
    classes=('unchange', 'change'),
    palette=[[0, 0, 0], [255, 255, 255]])

crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True

model = dict(
    type='ChangeCLIP',
    
    #pretrained='/home/dc001/.cache/clip/RN50.pt',   
    pretrained='/home/dc001/.cache/clip/ViT-L-14.pt', 
    #pretrained = '/home/dc001/.cache/clip/ViT-B-16.pt', 
    context_length=64,
    text_head=False,

    backbone=dict(
        type='PureCLIP',
        layers=[3, 4, 6, 3],
        #output_dim=1024,
        output_dim=768,
        input_resolution=512,
        #pretrained='/home/dc001/.cache/clip/RN50.pt',   
        pretrained='/home/dc001/.cache/clip/ViT-L-14.pt', 
        #pretrained = '/home/dc001/.cache/clip/ViT-B-16.pt',
        width=64,
        style='pytorch'),

    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=77,
        #embed_dim=1024,    
        embed_dim=768,  
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        context_length=16,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        #visual_dim=1024,   
        visual_dim=768,  
        dropout=0.1,
        #outdim=1024,
        outdim=768,        
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048, 4100],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='SwinTextDecode',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_classes=2,
        channels=256,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# 优化器（和 changeclip_clcd.py 保持一致）
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

# 数据加载
train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.txt'))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.txt'))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.txt'))

# 训练配置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))