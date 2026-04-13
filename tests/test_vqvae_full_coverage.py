from unittest.mock import Mock

import torch


def test_upsample_doubles_spatial_dimensions_and_runs_conv(upsample):
  """
  Verifies that the Upsample block correctly doubles the spatial dimensions 
  (height and width) of the input tensor. By using predefined weights (1.0)
  and zero bias, we can confidently assert the expected post-convolution matrix.
  """
  x = torch.ones(1, 1, 2, 2)

  out = upsample(x)

  expected = torch.tensor([[[[4.0, 6.0, 6.0, 4.0],
                             [6.0, 9.0, 9.0, 6.0],
                             [6.0, 9.0, 9.0, 6.0],
                             [4.0, 6.0, 6.0, 4.0]]]])
  assert out.shape == (1, 1, 4, 4)
  assert torch.allclose(out, expected)


def test_vector_quantizer_encode_decode_and_embed_roundtrip(vector_quantizer):
  """
  Validates the full lifecycle of the VectorQuantizer. 
  It sets up a mock embedding table manually and checks whether inputs 
  are correctly assigned to the closest centroids (indices) and if they 
  successfully reconstruct and fetch correct embeddings on the way out.
  """
  inputs = torch.tensor([[
    [0.1, 0.2],
    [2.2, 1.9],
    [-1.7, -2.1],
  ]])

  quantized, indices = vector_quantizer(inputs)
  decoded, decoded_indices = vector_quantizer.decode(indices)
  embedded = vector_quantizer.embed(torch.tensor([[1], [0], [2]]))

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


def test_encoder_decoder_forward_with_tiny_config(encoder, decoder, tiny_config, dummy_image_tensor):
  """
  Tests an end-to-end forward pass matching an Encoder connected to a Decoder 
  using randomly generated dummy image inputs. Checks that the encoded indices 
  respect the defined vocabulary constraints, shape boundaries, and that the 
  reconstructed output dimensions map faithfully to the original inputs while 
  staying numerically sound (no NaNs).
  """
  encoding_indices = encoder(dummy_image_tensor)
  decoded = decoder(encoding_indices)

  assert encoding_indices.shape == (2, tiny_config.quantized_resolution ** 2)
  assert encoding_indices.dtype == torch.int64
  assert int(encoding_indices.min()) >= 0
  assert int(encoding_indices.max()) < tiny_config.vocab_size
  assert decoder.last_z_shape == (
    2,
    tiny_config.z_channels,
    tiny_config.quantized_resolution,
    tiny_config.quantized_resolution,
  )
  assert decoded.shape == (2, tiny_config.out_channels, tiny_config.resolution, tiny_config.resolution)
  assert torch.isfinite(decoded).all()


def test_load_state_dict_from_url_delegates_to_torch_hub(monkeypatch, encoder, decoder):
  """
  Instead of hitting real URLs which would cause tests to fail without a network, 
  this mocks out `torch.hub.load_state_dict_from_url` and verifies whether 
  Encoder and Decoder delegate properly to load dict states using `weights_only=True` 
  for security and memory mapping to `cpu`, effectively testing the integration.
  """
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


def test_encoder_random_dimensions(encoder, tiny_config):
  """
  Tests boundary limits by feeding randomly shaped non-standard
  frames (that are divisible by the 16x downsample factor).
  The Encoder must be able to gracefully quantize them without crashing.
  """
  rand_h, rand_w = 16, 64
  
  x = torch.randn(1, tiny_config.in_channels, rand_h, rand_w)
  
  encoding_indices = encoder(x)
  
  expected_tokens = (rand_h // 2) * (rand_w // 2)
  
  assert encoding_indices.shape == (1, expected_tokens)
  assert encoding_indices.dtype == torch.int64
