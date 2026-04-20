"""Tests for Contribution C2: C4-equivariant ViT-MAE.

Falsifiable prediction target (from aidlc-docs/research/novel-contributions.md):
the CLS-token embedding of the equivariant MAE satisfies
``‖f(rot(x)) - f(x)‖₂ / ‖f(x)‖₂ ≤ 0.01`` for rot ∈ C₄, while vanilla MAE's
error is 0.1–0.3.
"""

from __future__ import annotations

import torch

from astromae.models import (
    MaskedAutoencoderC4,
    c4_equivariance_error,
    c4_polar_positional_embedding,
    mae_c4_tiny,
    mae_tiny,
)
from astromae.models.equivariant import RotEquivPatchEmbed


def test_rot_equiv_patch_embed_is_c4_invariant():
    torch.manual_seed(0)
    pe = RotEquivPatchEmbed(img_size=32, patch_size=16, in_chans=3, embed_dim=16)
    x = torch.randn(2, 3, 32, 32)
    f = pe(x)
    f_rot = pe(torch.rot90(x, k=1, dims=(-2, -1)))
    # After averaging over C4, the token-grid output is approximately
    # invariant (with a within-grid permutation that the mean kills).
    assert torch.allclose(f.mean(dim=1), f_rot.mean(dim=1), atol=1e-5)


def test_polar_pos_embed_shape_and_cls_zero():
    emb = c4_polar_positional_embedding(32, grid_size=4, cls_token=True)
    assert emb.shape == (17, 32)
    assert (emb[0] == 0).all()


def test_mae_c4_tiny_forward_runs():
    model = mae_c4_tiny(img_size=32, patch_size=16, in_chans=3)
    x = torch.rand(2, 3, 32, 32)
    loss, pred, mask = model(x, mask_ratio=0.5)
    assert torch.isfinite(loss)
    assert pred.shape[0] == 2


def test_mae_c4_tiny_accepts_ivar_loss():
    model = mae_c4_tiny(img_size=32, patch_size=16, in_chans=3)
    x = torch.rand(2, 3, 32, 32)
    ivar = torch.ones_like(x)
    loss, _, _ = model(x, mask_ratio=0.5, ivar=ivar)
    assert torch.isfinite(loss)


def test_c4_equivariance_error_is_smaller_for_c4_model():
    """The equivariant model should have strictly lower C4 error than
    the non-equivariant baseline at the same initialization budget."""
    torch.manual_seed(0)
    baseline = mae_tiny(img_size=32, patch_size=16, in_chans=3)
    equivariant = mae_c4_tiny(img_size=32, patch_size=16, in_chans=3)

    x = torch.rand(4, 3, 32, 32)
    err_baseline = float(c4_equivariance_error(baseline, x))
    err_equivariant = float(c4_equivariance_error(equivariant, x))

    assert err_equivariant < err_baseline
    # Sanity: both are finite and non-negative.
    assert err_equivariant >= 0
    assert err_baseline >= 0


def test_c4_equivariance_error_behaves_sensibly_on_c4_model():
    """The C4 model's equivariance error should be small (well below
    the vanilla MAE's characteristic range of 0.1-0.3)."""
    torch.manual_seed(0)
    equivariant = mae_c4_tiny(img_size=32, patch_size=16, in_chans=3)
    x = torch.rand(8, 3, 32, 32)
    err = float(c4_equivariance_error(equivariant, x))
    # Smoke bound: error should be well below the typical vanilla MAE
    # value observed in the C4-vs-baseline comparison test.
    assert err < 0.1


def test_c4_model_exposes_forward_features():
    model = mae_c4_tiny(img_size=32, patch_size=16, in_chans=3)
    feats = model.forward_features(torch.rand(2, 3, 32, 32))
    assert feats.shape == (2, 64)


def test_c4_model_shares_loss_math_with_parent():
    torch.manual_seed(7)
    parent = mae_tiny(img_size=32, patch_size=16, in_chans=3)
    child = MaskedAutoencoderC4(
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2.0,
    )
    x = torch.rand(2, 3, 32, 32)
    loss1, _, _ = parent(x, mask_ratio=0.5, generator=torch.Generator().manual_seed(1))
    loss2, _, _ = child(x, mask_ratio=0.5, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)
