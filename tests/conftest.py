"""Shared pytest setup for the V&V test suite."""

import sys
import torch
import pytest
from pathlib import Path

# Make the repo root importable so tests can do `from utils.vqvae import ...`
# without needing the package to be pip-installed.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from utils.vqvae import (
    CompressorConfig, ResnetBlock, AttnBlock, 
    Downsample, Upsample, VectorQuantizer, Encoder, Decoder
)

def pytest_configure(config):
  """Register custom markers so pytest doesn't warn about them."""
  config.addinivalue_line(
    "markers",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
  )

@pytest.fixture
def tiny_config():
    """Returns a minimal CompressorConfig for fast architecture testing."""
    return CompressorConfig(
        in_channels=3, out_channels=3, ch_mult=(1, 2),
        attn_resolutions=(4,), resolution=8, num_res_blocks=1,
        z_channels=32, vocab_size=8, ch=32, dropout=0.0,
    )

@pytest.fixture
def dummy_image_tensor(tiny_config):
    """Returns a synthetic image tensor (B, C, H, W)."""
    torch.manual_seed(0)
    return torch.randn(2, tiny_config.in_channels, tiny_config.resolution, tiny_config.resolution)

@pytest.fixture
def dummy_feature_tensor():
    """Returns a standard feature-map intermediate tensor (B=2, C=32, H=8, W=8)."""
    torch.manual_seed(0)
    return torch.randn(2, 32, 8, 8)

@pytest.fixture
def resnet_block():
    """Default ResnetBlock preserving shape/channels."""
    return ResnetBlock(in_channels=32, out_channels=32, dropout=0.0, temb_channels=64)

@pytest.fixture
def attn_block():
    """Default AttnBlock."""
    return AttnBlock(in_channels=32)

@pytest.fixture
def downsample():
    """Default Downsample block."""
    return Downsample(in_channels=16)

@pytest.fixture
def upsample():
    """Default Upsample block."""
    upsample = Upsample(in_channels=1)
    upsample.conv.weight.data.fill_(1.0)
    if upsample.conv.bias is not None:
        upsample.conv.bias.data.zero_()
    return upsample

@pytest.fixture
def vector_quantizer():
    """Pre-initialized VectorQuantizer with small dummy embeddings."""
    vq = VectorQuantizer(num_embeddings=3, embedding_dim=2)
    with torch.no_grad():
        vq._embedding.weight.copy_(
            torch.tensor([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]])
        )
    return vq

@pytest.fixture
def encoder(tiny_config):
    """Spatially scaled down Encoder."""
    return Encoder(tiny_config)

@pytest.fixture
def decoder(tiny_config):
    """Spatially scaled down Decoder."""
    return Decoder(tiny_config)
