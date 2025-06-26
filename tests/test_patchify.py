import torch
import pytest
from src.model import MaskedAutoencoder

def test_patchify_unpatchify():
    # Instantiate a small MAE model
    mae = MaskedAutoencoder(
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=32,
        depth=1,
        num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        mlp_ratio=2
    )
    
    # Create a dummy image tensor
    imgs = torch.rand(2, 3, 32, 32)
    
    # Apply patchify
    x = mae.patchify(imgs)
    
    # Check the shape of the patchified tensor
    # For 32x32 images with 16x16 patches, we should have 4 patches per image
    assert x.shape == (2, 4, 768), f"Expected shape (2, 4, 768), got {x.shape}"
    
    # Apply unpatchify
    recon = mae.unpatchify(x)
    
    # Check the shape of the reconstructed tensor
    assert recon.shape == imgs.shape, f"Expected shape {imgs.shape}, got {recon.shape}"
    
    # Assert that the reconstruction matches the input within tolerance
    assert torch.allclose(imgs, recon, atol=1e-6), "Reconstruction should match the input within tolerance"