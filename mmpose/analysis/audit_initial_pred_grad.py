import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint


CONFIG = (
    'projects/cigpose/body_2d_keypoint/'
    'cigpose-m_1xb384-420e_coco-256x192.py'
)

CHECKPOINT = '../checkpoints/cigpose-m_coco_256x192.pth'


def show_grad(name, param):
    print()
    print(name)
    print('  requires_grad:', param.requires_grad)

    if param.grad is None:
        print('  grad: None')
        return

    print(
        '  mean abs grad:',
        param.grad.abs().mean().item()
    )

    print(
        '  max abs grad:',
        param.grad.abs().max().item()
    )

    print(
        '  nonzero:',
        int(torch.count_nonzero(param.grad)),
        '/',
        param.grad.numel()
    )


def main():

    cfg = Config.fromfile(CONFIG)

    cfg.work_dir = (
        'work_dirs/challenge1_gradient_audit'
    )

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

    data_batch = next(
        iter(runner.train_dataloader)
    )

    data = model.data_preprocessor(
        data_batch,
        training=True
    )

    model.zero_grad(set_to_none=True)

    losses = model(
        **data,
        mode='loss'
    )

    print()
    print('Loss keys:', list(losses.keys()))

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

    print(
        'Total loss:',
        float(total_loss.detach().cpu())
    )

    total_loss.backward()

    print()
    print(
        '========== Gradient Audit =========='
    )

    # uncertainty / initial prediction branch
    show_grad(
        'initial_pred_x.weight',
        model.head.initial_pred_x.weight
    )

    show_grad(
        'initial_pred_x.bias',
        model.head.initial_pred_x.bias
    )

    show_grad(
        'initial_pred_y.weight',
        model.head.initial_pred_y.weight
    )

    show_grad(
        'initial_pred_y.bias',
        model.head.initial_pred_y.bias
    )

    # normal final prediction branch
    show_grad(
        'cls_gx.weight',
        model.head.cls_gx.weight
    )

    show_grad(
        'cls_gy.weight',
        model.head.cls_gy.weight
    )

    # canonical features
    show_grad(
        'canonical_features.weight',
        model.head.cim.canonical_features.weight
    )

    print()
    print(
        '===================================='
    )


if __name__ == '__main__':
    main()
