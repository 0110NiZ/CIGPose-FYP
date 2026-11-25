_base_ = '../current_batch128/config.py'
custom_imports = dict(imports=['kd'], allow_failed_imports=False)
model = dict(type='KDStudent', head=dict(type='KDHead'))
work_dir = '/root/autodl-tmp/cigpose-original/group_c_kd/results'
