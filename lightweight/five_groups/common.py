_base_ = '../group_c_kd/config.py'

custom_imports = dict(imports=['kd', 'five_group_model'], allow_failed_imports=False)
model = dict(type='FiveGroupStudent', feature_weight=0.1)
