import pytest
import torch

from utils.vqvae import Normalize, nonlinearity


def test_nonlinearity_matches_x_sigmoid_x():
  """
  nonlinearity() is the swish function: x * sigmoid(x).
  This test checks the math is correct and the shape stays the same.
  """
  torch.manual_seed(0)
  x = torch.randn(2, 3, 4, 5)

  out = nonlinearity(x)
  # This is the exact formula used in vqvae.py.
  expected = x * torch.sigmoid(x)

  assert out.shape == x.shape
  assert torch.allclose(out, expected)


def test_normalize_constructs_groupnorm_with_expected_hyperparams():
  """
  Normalize() should build a GroupNorm layer with the settings we expect.
  This is a simple constructor test (object-oriented check).
  """
  norm = Normalize(32)

  assert isinstance(norm, torch.nn.GroupNorm)
  assert norm.num_groups == 32
  assert norm.num_channels == 32
  assert norm.eps == pytest.approx(1e-6)
  assert norm.affine is True


def test_normalize_forward_preserves_shape_and_is_finite():
  """
  Normalize forward pass should not change the tensor shape.
  It should also not produce NaN or inf values.
  """
  torch.manual_seed(0)
  norm = Normalize(32)
  x = torch.randn(2, 32, 8, 8)

  out = norm(x)

  assert out.shape == x.shape
  assert torch.isfinite(out).all()


def test_encoder_whitebox_intermediate_shapes_via_hooks(encoder, tiny_config, dummy_image_tensor):
  """
  White-box test. We look inside Encoder and record shapes from a few layers.
  This helps catch silent shape bugs even if the forward pass still runs.

  We use forward hooks to capture outputs without changing the model code.
  """
  seen = {}

  def save_shape(name):
    def _hook(_module, _inputs, output):
      # Store the output shape for this layer name.
      seen[name] = tuple(output.shape)
    return _hook

  # Attach hooks to internal layers we care about.
  h1 = encoder.conv_in.register_forward_hook(save_shape("conv_in"))
  h2 = encoder.conv_out.register_forward_hook(save_shape("conv_out"))
  h3 = encoder.quant_conv.register_forward_hook(save_shape("quant_conv"))
  try:
    # Run a normal forward pass.
    encoding_indices = encoder(dummy_image_tensor)
  finally:
    # Always remove hooks so they do not leak into other tests.
    h1.remove()
    h2.remove()
    h3.remove()

  # With tiny_config (resolution=8), the encoder downsamples once.
  # So the spatial size at the end should be 4x4 (quantized_resolution).
  assert seen["conv_in"] == (2, tiny_config.ch, tiny_config.resolution, tiny_config.resolution)
  assert seen["conv_out"] == (2, tiny_config.z_channels, tiny_config.quantized_resolution, tiny_config.quantized_resolution)
  assert seen["quant_conv"] == (2, tiny_config.z_channels, tiny_config.quantized_resolution, tiny_config.quantized_resolution)

  # Output indices are flattened tokens. Token count should be h*w.
  assert encoding_indices.shape == (2, tiny_config.quantized_resolution * tiny_config.quantized_resolution)
  assert encoding_indices.dtype == torch.int64
