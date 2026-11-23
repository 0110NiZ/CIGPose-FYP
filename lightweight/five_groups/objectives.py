"""Train-time tensor operations shared by all five comparisons."""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


@contextmanager
def capture_output(module):
    outputs = []

    def save_output(_module, _args, output):
        outputs.append(output)

    handle = module.register_forward_hook(save_output)
    try:
        yield outputs
    finally:
        handle.remove()


def joint_cosine(student, teacher, weights):
    if student.shape != teacher.shape or student.shape[1:] != (17, 256):
        raise ValueError(f"Expected matching [B,17,256] tensors, got {student.shape}, {teacher.shape}")
    if weights.shape != student.shape[:2]:
        raise ValueError(f"Unexpected joint mask: {weights.shape}")
    distances = 1 - F.cosine_similarity(student.float(), teacher.detach().float(), dim=-1)
    valid = weights > 0
    return distances.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def spatial_cosine(projected, teacher):
    if projected.shape != teacher.shape or projected.shape[1:] != (768, 8, 6):
        raise ValueError(f"Expected matching [B,768,8,6] tensors, got {projected.shape}, {teacher.shape}")
    return (1 - F.cosine_similarity(projected.float(), teacher.detach().float(), dim=1)).mean()


def transfer_blocks(student, teacher):
    """Copy parameters and BN buffers once; strict matching catches incompatible weights."""
    for name in ('cim', 'gcn'):
        destination = getattr(student.head, name)
        source = getattr(teacher.head, name).state_dict()
        destination.load_state_dict(source, strict=True)
        if not all(torch.equal(v, source[k]) for k, v in destination.state_dict().items()):
            raise RuntimeError(f'Copy verification failed for head.{name}')


def reasoning_param_groups(optimizer, student):
    selected = {id(p) for m in (student.head.cim, student.head.gcn) for p in m.parameters()}
    groups = []
    found = set()
    for group in optimizer.param_groups:
        members = {id(p) for p in group['params']}
        overlap = selected & members
        if overlap:
            if overlap != members:
                raise ValueError("The optimizer mixes reasoning and other parameters in one group. "
                                 "Use the branch's original paramwise_cfg.")
            groups.append(group)
            found.update(overlap)
    if found != selected or not groups:
        raise ValueError('The optimizer must contain all CIM/GNN parameters before freezing.')
    return groups


def early_phase(student, policy, epoch_number):
    """Call after model.train() on the first batch of each epoch."""
    if policy not in ('freeze', 'slow') or epoch_number < 1:
        raise ValueError((policy, epoch_number))
    frozen = policy == 'freeze' and epoch_number <= 10
    for module in (student.head.cim, student.head.gcn):
        module.requires_grad_(not frozen)
        module.train(not frozen)
    if epoch_number > 10:
        return 1.
    return 0. if frozen else 0.1
