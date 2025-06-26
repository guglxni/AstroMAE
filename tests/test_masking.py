import torch
import pytest
from src.model import random_masking

def test_random_masking():
    # Create a dummy tensor
    x = torch.rand(2, 16, 768)
    
    # Apply random masking with 50% ratio
    x_masked, mask, ids_restore = random_masking(x, 0.5)
    
    # Assert the shapes are correct
    assert x_masked.shape == (2, 8, 768)
    assert mask.shape == (2, 16)
    
    # Each sample should mask exactly half the patches
    assert int(mask.sum().item()) == 16
    
    # Additional checks
    assert ids_restore.shape == (2, 16), "Restore indices should have shape (batch_size, seq_len)"
    
    # Check that mask is binary (contains only 0s and 1s)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0}), "Mask should only contain 0s and 1s"
    
    # Check that each sample has exactly 8 masked tokens (50% of 16)
    assert torch.all(mask.sum(dim=1) == 8), "Each sample should have exactly 8 masked tokens"