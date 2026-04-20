import pytest
import torch

from astromae.models.mae import MaskedAutoencoder, mae_tiny


@pytest.mark.parametrize("in_chans", [1, 3])
def test_patchify_unpatchify_roundtrip(in_chans):
    mae = MaskedAutoencoder(
        img_size=32,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=32,
        depth=1,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        mlp_ratio=2,
    )

    imgs = torch.rand(2, in_chans, 32, 32)

    x = mae.patchify(imgs)
    assert x.shape == (2, 4, 16 * 16 * in_chans)

    recon = mae.unpatchify(x)
    assert recon.shape == imgs.shape
    assert torch.allclose(imgs, recon, atol=1e-6)


def test_patchify_rejects_mismatched_image():
    mae = mae_tiny(img_size=32, patch_size=16, in_chans=3)
    with pytest.raises(ValueError):
        mae.patchify(torch.rand(1, 3, 30, 32))


def test_forward_produces_scalar_loss():
    mae = mae_tiny(img_size=32, patch_size=16, in_chans=3)
    imgs = torch.rand(2, 3, 32, 32)
    loss, pred, mask = mae(imgs, mask_ratio=0.5)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    assert pred.shape[0] == 2
    assert mask.shape == (2, (32 // 16) ** 2)


def test_forward_features_produces_cls_embedding():
    mae = mae_tiny(img_size=32, patch_size=16, in_chans=3)
    imgs = torch.rand(2, 3, 32, 32)
    feats = mae.forward_features(imgs)
    assert feats.shape == (2, 64)
