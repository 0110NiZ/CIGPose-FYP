import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


CONFIG = (
    'projects/cigpose/body_2d_keypoint/'
    'cigpose-m_1xb384-420e_coco-256x192.py'
)

CHECKPOINT = '../checkpoints/cigpose-m_coco_256x192.pth'


def clone_param(param):
    return param.detach().cpu().clone()


def report_change(name, before, after):
    diff = (after - before).abs()

    print()
    print(name)
    print('  max abs change :', diff.max().item())
    print('  mean abs change:', diff.mean().item())
    print(
        '  changed elems  :',
        int(torch.count_nonzero(diff).item()),
        '/',
        diff.numel()
    )


def main():
    cfg = Config.fromfile(CONFIG)

    cfg.work_dir = 'work_dirs/challenge1_update_audit'

    cfg.train_dataloader.batch_size = 2
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False

    runner = Runner.from_cfg(cfg)

    model = runner.model

    load_checkpoint(
        model,
        CHECKPOINT,
        map_location='cpu'
    )

    model.cuda()
    model.train()

    data_batch = next(iter(runner.train_dataloader))

    data = model.data_preprocessor(
        data_batch,
        training=True
    )

    # Save parameters before the update
    before = {
        'initial_pred_x.weight':
            clone_param(model.head.initial_pred_x.weight),

        'initial_pred_x.bias':
            clone_param(model.head.initial_pred_x.bias),

        'initial_pred_y.weight':
            clone_param(model.head.initial_pred_y.weight),

        'initial_pred_y.bias':
            clone_param(model.head.initial_pred_y.bias),

        # Positive controls
        'cls_gx.weight':
            clone_param(model.head.cls_gx.weight),

        'cls_gy.weight':
            clone_param(model.head.cls_gy.weight),

        'canonical_features.weight':
            clone_param(model.head.cim.canonical_features.weight),
    }

    # A normal optimizer step.
    # This only changes the model in memory.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0
    )

    optimizer.zero_grad(set_to_none=True)

    losses = model(
        **data,
        mode='loss'
    )

    total_loss = 0

    for name, value in losses.items():

        if 'loss' not in name:
            continue

        if isinstance(value, (list, tuple)):
            total_loss += sum(
                v.mean() for v in value
            )
        else:
            total_loss += value.mean()

    print()
    print('Loss keys:', list(losses.keys()))
    print(
        'Total loss:',
        float(total_loss.detach().cpu())
    )

    total_loss.backward()

    optimizer.step()

    after = {
        'initial_pred_x.weight':
            clone_param(model.head.initial_pred_x.weight),

        'initial_pred_x.bias':
            clone_param(model.head.initial_pred_x.bias),

        'initial_pred_y.weight':
            clone_param(model.head.initial_pred_y.weight),

        'initial_pred_y.bias':
            clone_param(model.head.initial_pred_y.bias),

        'cls_gx.weight':
            clone_param(model.head.cls_gx.weight),

        'cls_gy.weight':
            clone_param(model.head.cls_gy.weight),

        'canonical_features.weight':
            clone_param(model.head.cim.canonical_features.weight),
    }

    print()
    print('========== Parameter Update Audit ==========')

    for name in before:
        report_change(
            name,
            before[name],
            after[name]
        )

    print()
    print('============================================')


if __name__ == '__main__':
    main()
