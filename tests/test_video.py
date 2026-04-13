import numpy as np
import pytest

from utils.video import transform_img, transpose_and_clip

def test_transform_img_aspect_ratios():
    """
    Verifies that transform_img can handle various non-standard 
    input aspect ratios and dimensions (portrait, square, ultrawide).
    It must uniformly output images as (128, 256, 3) for the pipeline.
    """
    sizes = [
        (1080, 1920, 3), # Portrait HD
        (400, 400, 3),   # Square
        (150, 600, 3),   # Extremely wide
        (768, 512, 3),   # Random standard
    ]
    for size in sizes:
        # Create dummy image data
        frame = np.random.randint(0, 256, size, dtype=np.uint8)
        out = transform_img(frame)
        
        # Verify OpenCV output format shape
        assert out.shape == (128, 256, 3), (
            f"Expected shape (128, 256, 3) for input size {size}, got {out.shape}"
        )


def test_transpose_and_clip():
    """
    Verifies that we correctly reorder axes from PyTorch tensors (N, C, H, W) 
    back to standard frames (N, H, W, C) while safely clamping floating values 
    strictly between [0, 255] and returning them as uint8. 
    """
    # Create crazy tensors stretching outside bounds -100 to 300
    tensors = np.random.uniform(-100.0, 300.0, size=(2, 3, 10, 10))
    
    clipped = transpose_and_clip(tensors)
    
    # Check the axis reordering
    assert clipped.shape == (2, 10, 10, 3)
    
    # Check clipping constraints
    assert clipped.dtype == np.uint8
    assert int(clipped.min()) >= 0
    assert int(clipped.max()) <= 255
