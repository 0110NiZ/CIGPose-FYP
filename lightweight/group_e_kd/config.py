_base_ = '../group_c_kd/config.py'

custom_imports = dict(imports=['feature_kd'], allow_failed_imports=False)
model = dict(type='FeatureKDStudent', train_cfg=dict(feature_weight=5.0))
custom_hooks = _base_.custom_hooks + [
    dict(type='TransferAndFinetuneHook', warm_epochs=10, lr_factor=0.1),
]
val_evaluator = [_base_.val_evaluator, dict(type='OccludedJointError')]
work_dir = '../work_dirs/group_e_kd'
