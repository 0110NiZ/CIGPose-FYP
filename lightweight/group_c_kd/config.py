_base_ = '../configs/team128_baseline100.py'

custom_imports = dict(imports=['kd'], allow_failed_imports=False)
model = dict(type='KDStudent', head=dict(type='KDHead'),
             train_cfg=dict(kd_weight=0.04969))
work_dir = '../work_dirs/group_c_kd'
