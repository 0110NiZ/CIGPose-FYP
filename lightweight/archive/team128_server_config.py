auto_scale_lr = dict(enable=False, base_batch_size=128)
backend_args = dict(backend='local')
base_lr = 0.0003333333333333333
codec = dict(
    input_size=(
        192,
        256,
    ),
    normalize=False,
    sigma=(
        4.9,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=93,
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=60,
                scale_factor=[
                    0.75,
                    1.25,
                ],
                shift_factor=0.0,
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=0.5,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_mode = 'topdown'
data_root = '../data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        _scope_='mmpose',
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        _scope_='mmpose',
        interval=10,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(_scope_='mmpose', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmpose', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmpose', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmpose', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmpose', enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = (
    192,
    256,
)
keypoint_groups_definition = [
    [
        0,
        1,
        2,
        3,
        4,
    ],
    [
        5,
        7,
        9,
    ],
    [
        6,
        8,
        10,
    ],
    [
        11,
        12,
    ],
    [
        11,
        13,
        15,
    ],
    [
        12,
        14,
        16,
    ],
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmpose',
    by_epoch=True,
    num_digits=6,
    type='LogProcessor',
    window_size=50)
max_epochs = 100
model = dict(
    backbone=dict(
        _scope_='mmdet',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.33,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            '/root/autodl-tmp/cigpose-original/checkpoints/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        out_indices=(4, ),
        type='CSPNeXt',
        widen_factor=0.5),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        cim_cfg=dict(
            feature_dim=256,
            intervention_k=1,
            intervention_k_val=1,
            intervention_strategy='topk',
            num_keypoints=17),
        decoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        gcn_cfg=dict(hidden_dim=256, input_dim=256, num_layers=1),
        in_channels=512,
        in_featuremap_num=1,
        in_featuremap_size=(
            6,
            8,
        ),
        input_size=(
            192,
            256,
        ),
        keypoint_connections=[
            (
                0,
                1,
            ),
            (
                0,
                2,
            ),
            (
                1,
                2,
            ),
            (
                2,
                4,
            ),
            (
                3,
                1,
            ),
            (
                4,
                6,
            ),
            (
                3,
                5,
            ),
            (
                5,
                6,
            ),
            (
                6,
                8,
            ),
            (
                8,
                10,
            ),
            (
                5,
                7,
            ),
            (
                7,
                9,
            ),
            (
                5,
                11,
            ),
            (
                6,
                12,
            ),
            (
                11,
                12,
            ),
            (
                11,
                13,
            ),
            (
                13,
                15,
            ),
            (
                14,
                12,
            ),
            (
                14,
                16,
            ),
        ],
        keypoint_groups=[
            [
                0,
                1,
                2,
                3,
                4,
            ],
            [
                5,
                7,
                9,
            ],
            [
                6,
                8,
                10,
            ],
            [
                11,
                12,
            ],
            [
                11,
                13,
                15,
            ],
            [
                12,
                14,
                16,
            ],
        ],
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        loss_cf=dict(
            loss_weight=0.1, type='CounterfactualConsistencyLoss',
            use_kl=True),
        out_channels=17,
        simcc_split_ratio=2.0,
        type='CIGHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')
num_keypoints = 17
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0003333333333333333, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=50,
        begin=50,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=1.6666666666666667e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = False
skeleton = [
    (
        0,
        1,
    ),
    (
        0,
        2,
    ),
    (
        1,
        2,
    ),
    (
        2,
        4,
    ),
    (
        3,
        1,
    ),
    (
        4,
        6,
    ),
    (
        3,
        5,
    ),
    (
        5,
        6,
    ),
    (
        6,
        8,
    ),
    (
        8,
        10,
    ),
    (
        5,
        7,
    ),
    (
        7,
        9,
    ),
    (
        5,
        11,
    ),
    (
        6,
        12,
    ),
    (
        11,
        12,
    ),
    (
        11,
        13,
    ),
    (
        13,
        15,
    ),
    (
        14,
        12,
    ),
    (
        14,
        16,
    ),
]
stage2_num_epochs = 7
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        data_mode='topdown',
        data_prefix=dict(img='val2017/'),
        data_root='../data/coco/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='../data/coco/annotations/person_keypoints_val2017.json',
    type='CocoMetric')
train_batch_size = 128
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        ann_file=
        '/root/autodl-tmp/cigpose-original/experiments/small_backbone_30_seed21_precision3/person_keypoints_train2017_stratified30_seed21.json',
        data_mode='topdown',
        data_prefix=dict(img='train2017/'),
        data_root='../data/coco/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=80,
                scale_factor=[
                    0.6,
                    1.4,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=1.0,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        192,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        4.9,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=1.0,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                192,
                256,
            ),
            normalize=False,
            sigma=(
                4.9,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_batch_size = 64
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        data_mode='topdown',
        data_prefix=dict(img='val2017/'),
        data_root='../data/coco/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='../data/coco/annotations/person_keypoints_val2017.json',
    type='CocoMetric')
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(_scope_='mmpose', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmpose',
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/root/autodl-tmp/cigpose-original/experiments/small_backbone_30_seed21_precision3/work_dirs/team128_baseline100'

