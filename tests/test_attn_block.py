import torch

from utils.vqvae import AttnBlock


def test_attn_block_preserves_shape():
  """
  AttnBlock should preserve the input tensor's shape.
  """
  torch.manual_seed(0)

  block = AttnBlock(in_channels=32)

  x = torch.randn(2, 32, 8, 8)

  out = block(x)

  assert out.shape == x.shape


def test_attn_block_changes_output():
  """
  AttnBlock should modify the input via residual attention.
  """
  torch.manual_seed(0)

  block = AttnBlock(in_channels=32)

  x = torch.randn(2, 32, 8, 8)

  out = block(x)

  assert not torch.allclose(out, x)


def test_attn_block_different_channels():
  """
  AttnBlock should work with different channel counts.
  """
  torch.manual_seed(0)

  for ch in [32, 64, 128]:
    block = AttnBlock(in_channels=ch)
    x = torch.randn(1, ch, 4, 4)
    out = block(x)
    assert out.shape == x.shape
