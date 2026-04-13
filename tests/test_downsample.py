import torch

from utils.vqvae import Downsample

def test_downsample_even_input_reduces_spatial_dimensions(downsample):
    """
    Downsample should halve height and width for an even-sized input while
    preserving batch and channel dimensions.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 16, 8, 8)
    out = downsample(x)

    assert out.shape == (2, 16, 4, 4)
    assert out.dtype == x.dtype
    assert out.device == x.device

def test_downsample_odd_input_uses_asymmetric_padding():
    """
    Downsample should pad odd-sized inputs on the right and bottom before
    applying the stride-2 convolution.
    """
    down = Downsample(in_channels=1)
    down.conv.weight.data.fill_(1.0)
    if down.conv.bias is not None:
        down.conv.bias.data.zero_()

    x = torch.ones(1, 1, 5, 5)
    out = down(x)

    assert out.shape == (1, 1, 2, 2)
    assert torch.allclose(out, torch.full_like(out, 9.0))
