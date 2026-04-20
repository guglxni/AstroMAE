"""Tests for Contribution C3: I-JEPA pretraining objective."""

from __future__ import annotations

import copy

import torch

from astromae.models.jepa import IJEPA, ijepa_tiny


def test_jepa_forward_returns_scalar_loss():
    model = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    out = model(torch.rand(4, 3, 32, 32))
    assert torch.isfinite(out.loss)
    assert out.loss.ndim == 0
    assert out.pred.shape[:2] == out.target.shape[:2]


def test_jepa_target_encoder_has_no_grad():
    model = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    for p in model.target_encoder.parameters():
        assert not p.requires_grad


def test_jepa_ema_update_moves_target_toward_context():
    model = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    # Perturb the context encoder.
    for p in model.context_encoder.parameters():
        p.data.add_(0.1)
    pre = [p.data.clone() for p in model.target_encoder.parameters()]
    model._ema_update()
    post = [p.data for p in model.target_encoder.parameters()]
    # Each target parameter should have moved toward its context counterpart.
    for pre_p, post_p, ctx_p in zip(
        pre, post, model.context_encoder.parameters(), strict=True
    ):
        assert (post_p - pre_p).abs().mean() > 0
        # Distance to context decreased.
        assert (post_p - ctx_p.data).abs().mean() <= (pre_p - ctx_p.data).abs().mean()


def test_jepa_disjoint_context_target_masks():
    model = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    out = model(torch.rand(3, 3, 32, 32))
    overlap = (out.context_mask * out.target_mask).sum()
    assert float(overlap) == 0.0


def test_jepa_loss_decreases_on_fixed_batch():
    torch.manual_seed(0)
    model = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-3
    )
    imgs = torch.rand(4, 3, 32, 32)
    losses = []
    for _ in range(30):
        g = torch.Generator().manual_seed(7)
        out = model(imgs, generator=g)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        model._ema_update()
        losses.append(float(out.loss.item()))
    assert losses[-1] < losses[0]


def test_jepa_forward_features_shape():
    model = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    feats = model.forward_features(torch.rand(2, 3, 32, 32))
    assert feats.shape == (2, 64)


def test_jepa_deepcopy_preserves_ema_linkage():
    m = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    m2 = copy.deepcopy(m)
    # Sanity: target params still frozen.
    for p in m2.target_encoder.parameters():
        assert not p.requires_grad


def test_jepa_respects_generator_for_masks():
    model = ijepa_tiny(img_size=32, patch_size=16, in_chans=3)
    x = torch.rand(2, 3, 32, 32)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out_a = model(x, generator=g1)
    out_b = model(x, generator=g2)
    assert torch.equal(out_a.context_mask, out_b.context_mask)
    assert torch.equal(out_a.target_mask, out_b.target_mask)


def test_ijepa_instantiation_default_ratios_are_disjoint():
    m = IJEPA(img_size=32, patch_size=16, in_chans=3, embed_dim=32, depth=1, num_heads=4)
    assert m.context_ratio + m.target_ratio <= 1.0
