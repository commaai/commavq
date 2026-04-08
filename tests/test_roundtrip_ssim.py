"""
End-to-end encode -> decode round-trip validation via SSIM.

This is the most important validation test in the suite: it loads the real
trained checkpoints, runs known driving frames through the full compressor,
and asserts that the reconstructed frames are visually similar to the
originals (per Structural Similarity Index Measure).

We start simple: decode the pre-computed tokens in examples/tokens.npy and
compare against the source frames in examples/sample_video_ecamera.hevc.
The tokens were produced by the encoder from that exact video, so this is a
true round-trip even though we are not re-running the encoder here. Later
we can extend the file to also re-encode frames on the fly.

The test is marked `slow` and is skipped automatically when:
  - scikit-image is not installed
  - the decoder checkpoint cannot be downloaded
  - OpenCV cannot decode the sample HEVC clip
"""

from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = REPO_ROOT / "examples" / "sample_video_ecamera.hevc"
SAMPLE_TOKENS = REPO_ROOT / "examples" / "tokens.npy"

# Permissive starting threshold; tighten after a first real measurement.
SSIM_THRESHOLD = 0.6
# Keep the test fast: only round-trip a handful of frames.
NUM_FRAMES = 5

skimage = pytest.importorskip("skimage.metrics", reason="scikit-image not installed")


@pytest.fixture(scope="module")
def decoder():
  """Load the real Decoder with pretrained weights, on CPU."""
  from utils.vqvae import CompressorConfig, Decoder

  config = CompressorConfig()
  with torch.device("meta"):
    model = Decoder(config)
  try:
    model.load_state_dict_from_url(assign=True)
  except Exception as e:
    pytest.skip(f"could not download decoder checkpoint: {e}")
  return model.eval()


@pytest.fixture(scope="module")
def source_frames():
  """Read the first NUM_FRAMES frames of the sample clip and run them
  through transform_img so they line up with what the encoder originally saw."""
  from utils.video import read_video, transform_img

  if not SAMPLE_VIDEO.exists():
    pytest.skip(f"sample video missing: {SAMPLE_VIDEO}")
  try:
    video = read_video(str(SAMPLE_VIDEO))
  except Exception as e:
    pytest.skip(f"OpenCV could not decode sample video: {e}")

  frames = np.stack([transform_img(video[i]) for i in range(NUM_FRAMES)], axis=0)
  # frames is (NUM_FRAMES, 128, 256, 3), uint8, RGB
  assert frames.shape == (NUM_FRAMES, 128, 256, 3)
  return frames


@pytest.mark.slow
def test_roundtrip_ssim_above_threshold(decoder, source_frames):
  """Decode known tokens and assert per-frame SSIM is above the threshold."""
  from utils.video import transpose_and_clip

  if not SAMPLE_TOKENS.exists():
    pytest.skip(f"sample tokens missing: {SAMPLE_TOKENS}")

  tokens = np.load(SAMPLE_TOKENS).astype(np.int64)
  assert tokens.shape[1] == 128, f"unexpected token shape: {tokens.shape}"

  # Decode the same NUM_FRAMES frames in a single batched forward pass.
  token_batch = torch.from_numpy(tokens[:NUM_FRAMES])
  with torch.no_grad():
    decoded = decoder(token_batch).cpu().numpy()
  # decoded is (NUM_FRAMES, 3, 128, 256) float; convert to (N, H, W, C) uint8.
  decoded = transpose_and_clip(decoded)
  assert decoded.shape == source_frames.shape

  # Per-frame SSIM, averaged over the batch. channel_axis=-1 treats the RGB
  # channels jointly (one SSIM per frame, not per channel).
  ssims = [
    skimage.structural_similarity(
      source_frames[i], decoded[i], channel_axis=-1, data_range=255
    )
    for i in range(NUM_FRAMES)
  ]
  mean_ssim = float(np.mean(ssims))

  assert mean_ssim >= SSIM_THRESHOLD, (
    f"round-trip SSIM {mean_ssim:.3f} below threshold {SSIM_THRESHOLD}; "
    f"per-frame: {[round(s, 3) for s in ssims]}"
  )
