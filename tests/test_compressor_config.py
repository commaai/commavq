import pytest
import torch
from utils.vqvae import CompressorConfig

def test_compressor_config_defaults():
    """
    simple test for CompressorConfig
    """
    config = CompressorConfig()
    
    # Test default properties
    assert config.in_channels == 3
    assert config.out_channels == 3
    assert config.resolution == 256
    assert config.vocab_size == 1024
    assert config.ch_mult == (1, 1, 2, 2, 4)
    
    # Test computed properties
    assert config.num_resolutions == 5  
    
    # quantized_resolution = resolution // 2**(num_resolutions-1)
    # 256 // 2**(5-1) = 256 // 16 = 16
    assert config.quantized_resolution == 16
