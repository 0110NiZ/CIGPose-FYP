from mmengine.config import Config
from mmengine.runner import Runner


CONFIG = 'projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py'


def show_obj(name, obj):
    print(f'\n========== {name} ==========')
    print('type:', type(obj))

    if hasattr(obj, 'keys'):
        try:
            print('keys:', list(obj.keys()))
        except Exception:
            pass

    if hasattr(obj, 'shape'):
        print('shape:', obj.shape)

    print(obj)


def main():
    cfg = Config.fromfile(CONFIG)
    cfg.work_dir = 'work_dirs/challenge1_inspect'

    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    runner = Runner.from_cfg(cfg)

    dataloader = runner.test_dataloader
    data_batch = next(iter(dataloader))

    print('\n==============================')
    print('RAW DATA BATCH')
    print('==============================')
    print('batch type:', type(data_batch))

    if isinstance(data_batch, dict):
        print('batch keys:', data_batch.keys())

    samples = data_batch['data_samples']
    sample = samples[0]

    print('\n==============================')
    print('FIRST DATA SAMPLE')
    print('==============================')

    print('sample type:', type(sample))

    try:
        print('sample keys:', list(sample.keys()))
    except Exception:
        pass

    print('\n----- METAINFO -----')
    try:
        for k, v in sample.metainfo.items():
            if k in [
                'img_id',
                'img_path',
                'bbox',
                'bbox_score',
                'input_center',
                'input_scale',
                'input_size',
                'ori_shape'
            ]:
                print(k, ':', v)
    except Exception as e:
        print('metainfo error:', e)

    print('\n----- gt_instances -----')
    if hasattr(sample, 'gt_instances'):
        gt = sample.gt_instances

        try:
            print('gt_instances keys:', list(gt.keys()))
        except Exception:
            pass

        for name in [
            'keypoints',
            'keypoints_visible',
            'bboxes',
            'bbox_scores',
            'bbox'
        ]:
            if hasattr(gt, name):
                show_obj(f'gt_instances.{name}', getattr(gt, name))
    else:
        print('No gt_instances')

    print('\n----- gt_instance_labels -----')
    if hasattr(sample, 'gt_instance_labels'):
        labels = sample.gt_instance_labels

        try:
            print('gt_instance_labels keys:', list(labels.keys()))
        except Exception:
            pass

        for name in [
            'keypoint_x_labels',
            'keypoint_y_labels',
            'keypoint_weights'
        ]:
            if hasattr(labels, name):
                show_obj(
                    f'gt_instance_labels.{name}',
                    getattr(labels, name)
                )
    else:
        print('No gt_instance_labels')


if __name__ == '__main__':
    main()
