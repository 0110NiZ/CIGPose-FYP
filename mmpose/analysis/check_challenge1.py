import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


CONFIG = 'projects/cigpose/body_2d_keypoint/cigpose-m_1xb384-420e_coco-256x192.py'
CHECKPOINT = '../checkpoints/cigpose-m_coco_256x192.pth'


def main():
    cfg = Config.fromfile(CONFIG)
    cfg.work_dir = 'work_dirs/challenge1_sanity'

    # Only need one sample for sanity check
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False

    runner = Runner.from_cfg(cfg)

    model = runner.model
    load_checkpoint(model, CHECKPOINT, map_location='cpu')

    model.cuda()
    model.eval()

    dataloader = runner.test_dataloader
    data_batch = next(iter(dataloader))

    with torch.no_grad():
        data = model.data_preprocessor(data_batch, training=False)

        inputs = data['inputs']
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cuda()

        feats = model.extract_feat(inputs)

        # Official CIGPose inference
        official_x, official_y = model.head.forward(feats)

        # Our Challenge-1 diagnostic path
        diag = model.head.diagnostic_forward(feats)

        top1 = int(diag['top1_indices'][0].item())

        # Our forced prediction for the exact joint selected by official rule
        forced_x, forced_y = diag['forced'][top1]

        diff_x = (official_x - forced_x).abs().max().item()
        diff_y = (official_y - forced_y).abs().max().item()

        confidence = diag['confidence'][0]
        confound = diag['confound_score'][0]

        print()
        print('========== Challenge 1 Sanity Check ==========')
        print(f'Official Top-1 joint id : {top1}')
        print(f'Confidence              : {confidence[top1].item():.8f}')
        print(f'Confound score           : {confound[top1].item():.8f}')
        print()
        print(f'Max |official_x - forced_x| = {diff_x:.12f}')
        print(f'Max |official_y - forced_y| = {diff_y:.12f}')
        print()

        tol = 1e-6

        if diff_x < tol and diff_y < tol:
            print('PASS: forced Top-1 matches official CIGPose Top-1.')
        else:
            print('FAIL: forced Top-1 does NOT match official CIGPose.')
            print('Do not continue to the full diagnostic experiment yet.')

        print('===============================================')


if __name__ == '__main__':
    main()
