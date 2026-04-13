import torch

from utils.vqvae import AttnBlock

def test_attn_block_preserves_shape(attn_block, dummy_feature_tensor):
  """
  AttnBlock should preserve the input tensor's shape.
  """
  out = attn_block(dummy_feature_tensor)
  assert out.shape == dummy_feature_tensor.shape

def test_attn_block_changes_output(attn_block, dummy_feature_tensor):
  """
  AttnBlock should modify the input via residual attention.
  """
  out = attn_block(dummy_feature_tensor)
  assert not torch.allclose(out, dummy_feature_tensor)

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
