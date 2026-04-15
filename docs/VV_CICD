CI/CD and Testing Documentation: commaVQ
Overveiw:
The commaVQ repository facilitates the compression of driving scenes into tokens for use in world models.To maintain the integrity of these models we have implemented an automated pipeline that validates every change.
CI/CD Pipeline Integration
TO make sure every change is set to the standards of Verification and Validation, we used GitHub Actions workflow
Goal: Confirm the test plan, architecture, and interface contracts are internally consistent before code execution

1.1 Decoder Forward-Pass Pipeline
The decoder must follow a strict sequential order to reconstruct images from encoding_indices. A failure in this ordering leads to catastrophic reconstruction error.

Execution Order:

Quantize: (B, 128) IDs → (B, 128, 256) embeddings.

Reshape: Rearrange to (B, 256, 8, 16).

Refine: post_quant_conv (1x1) → conv_in (3x3).

Process: Mid-stack (Resnet + Attention).

Upsample: Four stages (8x16 → 128x256).

Finalize: GroupNorm → Swish → Conv2d (3x3) → Scale to [0, 255].

Pipeline Architecture
Review 1.1.1 Pipeline diagram in docs/vv_plan.md

1.2 Shape Algebra Verification
Dimensions are tracked at each bottleneck to ensure no data is dropped or misaligned.
Stage
Expected Shape
Input Indices
(B, 128)
Latent Space
(B, 256, 8, 16)
Upsample 1
(B, *, 16, 32)
Upsample 4
(B, *, 128, 256)
Final RGB
(B, 3, 128, 256)


Phase 2 — Implementation Validation (Executable Tests)

Goal: Convert design contracts into automated pytest assertions.

2.1 Test Suite Structure
Tests are compartmentalized to isolate model logic from video utility logic.
test_decoder_forward.py: Validates shapes and tensor ranges.
test_upsample.py: Ensures the $2 \times$ spatial expansion logic is mathematically sound.
test_video_utils.py: Checks for color space consistency (BGR vs RGB).
test_roundtrip_ssim.py: The "Slow" end-to-end quality check.
2.2 Critical Test Definitions

SSIM Roundtrip: We perform an Encode → Decode cycle on real driving frames.
Success Metric: $SSIM \geq 0.6$ (Initial Threshold).
Significance: This is the ultimate proof that the compressor preserves the visual features necessary for world modeling.
Color Inversion Check: Validates that write_video correctly inverts the read_video color space. Without this, the model "sees" inverted colors, leading to catastrophic failure in downstream driving predictions.



3. Summary of Progress (What has been done)
We have successfully established the following:

Architecture Alignment: Confirmed that the Decoder pipeline in utils/vqvae.py matches the design specs for 4-stage upsampling.

Infrastructure Setup: Created the tests/ directory and conftest.py to provide stable mock data, preventing the need for large downloads during unit testing.

CI/CD Foundation: Configured the environment to run pytest automatically on push.

Utility Hardening: Verified that transform_img correctly normalizes varying input aspect ratios (e.g., 1080p, 720p) into the required (128, 256) resolution.
