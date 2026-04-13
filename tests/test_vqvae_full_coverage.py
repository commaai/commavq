from unittest.mock import Mock

import torch

from utils.vqvae import (
  CompressorConfig,
  Decoder,
  Encoder,
  Upsample,
  VectorQuantizer,
)


def make_tiny_config():
  """
  Creates a minimal CompressorConfig for testing purposes.
  By scaling down dimensions like resolution, z_channels, and vocab_size,
  the tests can run extremely quickly while verifying the architecture.
  """
  return CompressorConfig(
    in_channels=3,
    out_channels=3,
    ch_mult=(1, 2),
    attn_resolutions=(4,),
    resolution=8,
    num_res_blocks=1,
    z_channels=32,
    vocab_size=8,
    ch=32,
    dropout=0.0,
  )


def test_upsample_doubles_spatial_dimensions_and_runs_conv():
  """
  Verifies that the Upsample block correctly doubles the spatial dimensions 
  (height and width) of the input tensor. By using predefined weights (1.0)
  and zero bias, we can confidently assert the expected post-convolution matrix.
  """
  upsample = Upsample(in_channels=1)
  upsample.conv.weight.data.fill_(1.0)
  if upsample.conv.bias is not None:
    upsample.conv.bias.data.zero_()

  x = torch.ones(1, 1, 2, 2)

  out = upsample(x)

  expected = torch.tensor([[[[4.0, 6.0, 6.0, 4.0],
                             [6.0, 9.0, 9.0, 6.0],
                             [6.0, 9.0, 9.0, 6.0],
                             [4.0, 6.0, 6.0, 4.0]]]])
  assert out.shape == (1, 1, 4, 4)
  assert torch.allclose(out, expected)


def test_vector_quantizer_encode_decode_and_embed_roundtrip():
  """
  Validates the full lifecycle of the VectorQuantizer. 
  It sets up a mock embedding table manually and checks whether inputs 
  are correctly assigned to the closest centroids (indices) and if they 
  successfully reconstruct and fetch correct embeddings on the way out.
  """
  quantizer = VectorQuantizer(num_embeddings=3, embedding_dim=2)
  with torch.no_grad():
    quantizer._embedding.weight.copy_(
      torch.tensor([
        [0.0, 0.0],
        [2.0, 2.0],
        [-2.0, -2.0],
      ])
    )

  inputs = torch.tensor([[
    [0.1, 0.2],
    [2.2, 1.9],
    [-1.7, -2.1],
  ]])

  quantized, indices = quantizer(inputs)
  decoded, decoded_indices = quantizer.decode(indices)
  embedded = quantizer.embed(torch.tensor([[1], [0], [2]]))

  assert torch.equal(indices, torch.tensor([[0, 1, 2]]))
  assert torch.equal(decoded_indices, indices)
  assert torch.allclose(
    quantized,
    torch.tensor([[
      [0.0, 0.0],
      [2.0, 2.0],
      [-2.0, -2.0],
    ]]),
  )
  assert torch.allclose(decoded, quantized)
  assert torch.allclose(
    embedded,
    torch.tensor([
      [2.0, 2.0],
      [0.0, 0.0],
      [-2.0, -2.0],
    ]),
  )


def test_encoder_decoder_forward_with_tiny_config():
  """
  Tests an end-to-end forward pass matching an Encoder connected to a Decoder 
  using randomly generated dummy image inputs. Checks that the encoded indices 
  respect the defined vocabulary constraints, shape boundaries, and that the 
  reconstructed output dimensions map faithfully to the original inputs while 
  staying numerically sound (no NaNs).
  """
  torch.manual_seed(0)
  config = make_tiny_config()
  encoder = Encoder(config)
  decoder = Decoder(config)
  x = torch.randn(2, config.in_channels, config.resolution, config.resolution)

  encoding_indices = encoder(x)
  decoded = decoder(encoding_indices)

  assert encoding_indices.shape == (2, config.quantized_resolution ** 2)
  assert encoding_indices.dtype == torch.int64
  assert int(encoding_indices.min()) >= 0
  assert int(encoding_indices.max()) < config.vocab_size
  assert decoder.last_z_shape == (
    2,
    config.z_channels,
    config.quantized_resolution,
    config.quantized_resolution,
  )
  assert decoded.shape == (2, config.out_channels, config.resolution, config.resolution)
  assert torch.isfinite(decoded).all()


def test_load_state_dict_from_url_delegates_to_torch_hub(monkeypatch):
  """
  Instead of hitting real URLs which would cause tests to fail without a network, 
  this mocks out `torch.hub.load_state_dict_from_url` and verifies whether 
  Encoder and Decoder delegate properly to load dict states using `weights_only=True` 
  for security and memory mapping to `cpu`, effectively testing the integration.
  """
  config = make_tiny_config()
  encoder = Encoder(config)
  decoder = Decoder(config)
  encoder_state_dict = {"encoder_weight": torch.tensor([1.0])}
  decoder_state_dict = {"decoder_weight": torch.tensor([2.0])}

  load_from_url = Mock(side_effect=[encoder_state_dict, decoder_state_dict])
  encoder_load_state_dict = Mock()
  decoder_load_state_dict = Mock()

  monkeypatch.setattr(torch.hub, "load_state_dict_from_url", load_from_url)
  monkeypatch.setattr(encoder, "load_state_dict", encoder_load_state_dict)
  monkeypatch.setattr(decoder, "load_state_dict", decoder_load_state_dict)

  encoder.load_state_dict_from_url("https://example.com/encoder.bin", strict=False)
  decoder.load_state_dict_from_url("https://example.com/decoder.bin", assign=True)

  assert load_from_url.call_args_list == [
    (( "https://example.com/encoder.bin",), {"map_location": "cpu", "weights_only": True}),
    (( "https://example.com/decoder.bin",), {"map_location": "cpu", "weights_only": True}),
  ]
  encoder_load_state_dict.assert_called_once_with(encoder_state_dict, strict=False)
  decoder_load_state_dict.assert_called_once_with(decoder_state_dict, assign=True)
