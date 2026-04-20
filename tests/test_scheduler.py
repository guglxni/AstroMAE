import torch

from astromae.training.scheduler import CosineWarmupScheduler


def test_warmup_then_cosine_decay():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sch = CosineWarmupScheduler(opt, warmup_steps=5, total_steps=20, min_lr_ratio=0.1)

    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()

    assert lrs[0] < lrs[4]
    assert lrs[4] > lrs[-1]
    assert abs(lrs[-1] - 0.1) < 0.05


def test_zero_warmup_is_valid():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    CosineWarmupScheduler(opt, warmup_steps=0, total_steps=10)
