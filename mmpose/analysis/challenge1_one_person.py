import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


CONFIG = 'projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py'
CHECKPOINT = '../checkpoints/cigpose-m_coco_256x192.pth'

JOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]


def simcc_to_original(pred_x, pred_y, sample, split_ratio):
    """Decode SimCC logits and map coordinates back to original image."""

    # [B, K, W/H] -> [B, K]
    x_idx = pred_x.argmax(dim=-1).float()
    y_idx = pred_y.argmax(dim=-1).float()

    # SimCC index -> network input coordinates
    x_input = x_idx / split_ratio
    y_input = y_idx / split_ratio

    input_size = np.asarray(
        sample.metainfo['input_size'], dtype=np.float32)
    input_center = np.asarray(
        sample.metainfo['input_center'], dtype=np.float32)
    input_scale = np.asarray(
        sample.metainfo['input_scale'], dtype=np.float32)

    x_orig = (
        x_input.cpu().numpy() / input_size[0] * input_scale[0]
        + input_center[0] - input_scale[0] / 2.0
    )

    y_orig = (
        y_input.cpu().numpy() / input_size[1] * input_scale[1]
        + input_center[1] - input_scale[1] / 2.0
    )

    coords = np.stack([x_orig, y_orig], axis=-1)

    return coords


def normalized_errors(pred, gt, visible, norm):
    """Return per-joint normalized Euclidean error."""

    err = np.linalg.norm(pred - gt, axis=-1) / norm

    err[~visible] = np.nan

    return err


def main():
    cfg = Config.fromfile(CONFIG)
    cfg.work_dir = 'work_dirs/challenge1_one_person'

    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    runner = Runner.from_cfg(cfg)

    model = runner.model
    load_checkpoint(model, CHECKPOINT, map_location='cpu')

    model.cuda()
    model.eval()

    data_batch = next(iter(runner.test_dataloader))

    # Keep original data sample for GT/metainfo
    sample = data_batch['data_samples'][0]

    gt = sample.gt_instances.keypoints[0].astype(np.float32)

    visible = (
        sample.gt_instances.keypoints_visible[0] > 0
    )

    bbox = sample.gt_instances.bboxes[0].astype(np.float32)

    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Person-normalized distance
    norm = np.sqrt(bbox_w * bbox_h)

    with torch.no_grad():
        data = model.data_preprocessor(
            data_batch, training=False)

        inputs = data['inputs']

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda()

        feats = model.extract_feat(inputs)

        diag = model.head.diagnostic_forward(feats)

    split_ratio = model.head.simcc_split_ratio

    # No-intervention baseline
    base_x, base_y = diag['baseline']

    base_coords = simcc_to_original(
        base_x,
        base_y,
        sample,
        split_ratio,
    )[0]

    base_errors = normalized_errors(
        base_coords,
        gt,
        visible,
        norm,
    )

    base_pose_error = np.nanmean(base_errors)

    confound = (
        diag['confound_score'][0]
        .detach()
        .cpu()
        .numpy()
    )

    rows = []

    for joint_id, (corr_x, corr_y) in enumerate(diag['forced']):

        corr_coords = simcc_to_original(
            corr_x,
            corr_y,
            sample,
            split_ratio,
        )[0]

        corr_errors = normalized_errors(
            corr_coords,
            gt,
            visible,
            norm,
        )

        corr_pose_error = np.nanmean(corr_errors)

        pose_gain = (
            base_pose_error
            - corr_pose_error
        )

        if visible[joint_id]:
            base_local_error = base_errors[joint_id]
            corr_local_error = corr_errors[joint_id]

            local_gain = (
                base_local_error
                - corr_local_error
            )
        else:
            base_local_error = np.nan
            corr_local_error = np.nan
            local_gain = np.nan

        rows.append({
            'joint_id': joint_id,
            'joint_name': JOINT_NAMES[joint_id],
            'visible': int(visible[joint_id]),
            'confound_score': float(confound[joint_id]),
            'base_local_error': base_local_error,
            'corr_local_error': corr_local_error,
            'local_gain': local_gain,
            'base_pose_error': base_pose_error,
            'corr_pose_error': corr_pose_error,
            'pose_gain': pose_gain,
        })

    # Sort by uncertainty/confound score
    rows_u = sorted(
        rows,
        key=lambda x: x['confound_score'],
        reverse=True,
    )

    print()
    print('Image ID:', sample.metainfo['img_id'])
    print('BBox:', bbox)
    print('Visible joints:', int(visible.sum()), '/ 17')
    print()

    print(
        f'{"id":<3} '
        f'{"joint":<16} '
        f'{"vis":<4} '
        f'{"U":>10} '
        f'{"local_gain":>14} '
        f'{"pose_gain":>14}'
    )

    print('-' * 70)

    for r in rows_u:
        local = (
            f'{r["local_gain"]:+.6f}'
            if not np.isnan(r['local_gain'])
            else 'N/A'
        )

        print(
            f'{r["joint_id"]:<3} '
            f'{r["joint_name"]:<16} '
            f'{r["visible"]:<4} '
            f'{r["confound_score"]:>10.6f} '
            f'{local:>14} '
            f'{r["pose_gain"]:>+14.6f}'
        )

    uncertainty_top1 = max(
        rows,
        key=lambda x: x['confound_score'],
    )

    oracle_top1 = max(
        rows,
        key=lambda x: x['pose_gain'],
    )

    print()

    sorted_u = np.sort(confound)[::-1]

    print('========== Uncertainty Diagnostics ==========')
    print(f'Max U       : {confound.max():.9f}')
    print(f'Min U       : {confound.min():.9f}')
    print(f'Mean U      : {confound.mean():.9f}')
    print(f'Std U       : {confound.std():.9f}')
    print(f'Top1-Top2   : {(sorted_u[0] - sorted_u[1]):.9f}')
    print()

    print('========== Challenge 1 ==========')

    print(
        'Uncertainty Top-1:',
        uncertainty_top1['joint_id'],
        uncertainty_top1['joint_name'],
        'U =',
        f'{uncertainty_top1["confound_score"]:.6f}',
        'Pose Gain =',
        f'{uncertainty_top1["pose_gain"]:+.6f}',
    )

    print(
        'Oracle Gain Top-1:',
        oracle_top1['joint_id'],
        oracle_top1['joint_name'],
        'U =',
        f'{oracle_top1["confound_score"]:.6f}',
        'Pose Gain =',
        f'{oracle_top1["pose_gain"]:+.6f}',
    )

    print()

    if uncertainty_top1['joint_id'] == oracle_top1['joint_id']:
        print(
            'For this person: uncertainty Top-1 '
            'matches Oracle Gain Top-1.'
        )
    else:
        print(
            'For this person: uncertainty Top-1 '
            'does NOT match Oracle Gain Top-1.'
        )


if __name__ == '__main__':
    main()
