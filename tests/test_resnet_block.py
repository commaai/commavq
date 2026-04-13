import torch

from utils.vqvae import ResnetBlock

def test_resnet_block_preserves_shape_and_uses_temb(resnet_block, dummy_feature_tensor):
  """
  ResnetBlock should preserve spatial/channel shape when in/out channels match,
  and a provided timestep embedding should influence the output.
  """
  torch.manual_seed(0)

  temb = torch.randn(2, 64)

  out_with_temb = resnet_block(dummy_feature_tensor, temb)
  out_without_temb = resnet_block(dummy_feature_tensor, None)

  assert out_with_temb.shape == dummy_feature_tensor.shape
  assert out_without_temb.shape == dummy_feature_tensor.shape
  assert not torch.allclose(out_with_temb, out_without_temb)


def test_resnet_block_channel_change_supports_both_shortcuts(dummy_feature_tensor):
  """
  When channels change, ResnetBlock should project the residual path and return
  tensors with the requested out_channels for both shortcut variants.
  """
  nin_block = ResnetBlock(
    in_channels=32,
    out_channels=64,
    conv_shortcut=False,
    dropout=0.0,
    temb_channels=0,
  )
  assert hasattr(nin_block, "nin_shortcut")
  assert not hasattr(nin_block, "conv_shortcut")
  nin_out = nin_block(dummy_feature_tensor, None)
  assert nin_out.shape == (2, 64, 8, 8)

  conv_block = ResnetBlock(
    in_channels=32,
    out_channels=64,
    conv_shortcut=True,
    dropout=0.0,
    temb_channels=0,
  )
  assert hasattr(conv_block, "conv_shortcut")
  assert not hasattr(conv_block, "nin_shortcut")
  conv_out = conv_block(dummy_feature_tensor, None)
  assert conv_out.shape == (2, 64, 8, 8)
