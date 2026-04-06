# dataset settings
# dataset_type = 'TXTCDDataset'# mmseg\datasets\basetxtdataset.py
dataset_type = 'TXTCDDatasetJSON'# mmseg\datasets\basetxtdataset.py
data_root = '/home/dc001/data'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    # dict(type='MultiImgExchangeTime', prob=0.5),
    # ————2026-3-21——————修改——————见——————ChangeCLIP模型配置（SYSU-CD数据集）
    #dict(type='RandomRotFlip'), # 旋转 + 翻转
    #dict(type='PhotoMetricDistortion'), # 颜色抖动 (模拟光照变化)
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), # 避免采样全背景
    #dict(type='Pad', size=crop_size, pad_val=0),
    #dict(type='PackSegInputs'),
    # ————2026-3-21——————修改——————见——————ChangeCLIP模型配置（SYSU-CD数据集）
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='ConcatCDInput'),## 这个操作会把 image1 和 image2 在 channel 维度拼接 → [B, 6, H, W]
    dict(type='PackCDInputs')
]
val_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackCDInputs')
]
test_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackCDInputs')
]

tta_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[dict(type='LoadAnnotations')],
                    [dict(type='ConcatCDInput')],
                    [dict(type='PackCDInputs')]])
]
train_dataloader = dict(
    batch_size=4,
    #num_workers=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            #img_path='train/A',#LEVIR-CD
            #img_path='train/time1',#SYSU-CD
            img_path='train/image1',#CLCD,WHU-CD
            #img_path='train/t1',#DSIFN
            #img_path2='train/B',#LEVIR-CD
            #img_path2='train/time2',#SYSU-CD
            img_path2='train/image2',#CLCD,WHU-CD
            #img_path2='train/t2',#DSIFN
            seg_map_path='train/label'),#LEVIR-CD,SYSU-CD,CLCD,WHU-CD
            #seg_map_path='train/mask'),#DSIFN
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    #num_workers=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            #img_path='val/A', img_path2='val/B', seg_map_path='val/label'),#LEVIR-CD
            #img_path='val/time1', img_path2='val/time2', seg_map_path='val/label'),#SYSU-CD
            img_path='val/image1', img_path2='val/image2', seg_map_path='val/label'),#CLCD,WHU-CD
            #img_path='val/t1', img_path2='val/t2', seg_map_path='val/mask'),#DSIFN
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    #num_workers=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            #img_path='test/A', img_path2='test/B', seg_map_path='test/label'),#LEVIR-CD
            #img_path='test/time1', img_path2='test/time2', seg_map_path='test/label'),#SYSU-CD
            img_path='test/image1', img_path2='test/image2', seg_map_path='test/label'),#CLCD,WHU-CD
            #img_path='test/t1', img_path2='test/t2', seg_map_path='test/mask'),#DSIFN
        pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']  )

test_evaluator = val_evaluator
