# commavq
commaVQ is a  dataset of 100,000 heavily compressed driving videos,

| Real Video    | Compressed Video | Future Prediction |
| ------------- | ---------------- |------------------ |
| https://github.com/commaai/commavq/assets/29985433/91894bf7-592b-4204-b3f2-3e805984045c  | https://github.com/commaai/commavq/assets/29985433/3a799ac8-781e-461c-bf14-c15cea42b985    | https://github.com/commaai/commavq/assets/29985433/f6f7699b-b6cb-4f9c-80c9-8e00d75fbfae |


## GPT latency challenge: make me faster! $1000
Make the gpt model run faster on a consumer GPU (e.g. NVIDIA 3090)
Go to `./notebooks/gpt.ipynb` to start

| Implementation                                                                     | Latency       |
| :----------------------------------------------------------------------------------| ------------: |
| pytorch gpt-fast                                                                   | 0.3 sec/frame |
| onnxruntime-gpu with microsoft.Attention operators and past_present_share_buffer   | 0.4 sec/frame |
| onnxruntime-gpu with microsoft.Attention operators                                 | 0.5 sec/frame |
| Naive onnxruntime-gpu implementation                                               | 1.5 sec/frame |


## Compression challenge: make me smaller! $1000
losslessly compress one segment's tokens.
Go to `./notebooks/compress.ipynb` to start

| Implementation                                                                     | Compression rate |
| :----------------------------------------------------------------------------------| ---------------: |
| lzma                                                                               |  1.63            |


## Overview
A VQ-VAE [1,2] was used to heavily compress each frame into 128 "tokens" of 10 bits each. Each entry of the dataset is a "segment" of compressed driving video, i.e. 1min of frames at 20 FPS. Each file is of shape 1200x8x16 and saved as int16.

Note that the compressor is extremely lossy on purpose. It makes the dataset smaller and easy to play with (train GPT as a world model with large context size [3], fast autoregressive generation, etc.). We might extend the dataset to a less lossy version when we see fit.

## Examples
Checkout `./notebooks/encode.ipynb` and `./notebooks/decode.ipynb` for an example of how to visualize the dataset using a segment of driving video from [comma's drive to Taco Bell](https://blog.comma.ai/taco-bell/)

Checkout `./notebooks/gpt.ipynb` for an example of how to use a pretrained GPT model to imagine future frames.

Checkout `./notebooks/compress.ipynb` for an example of how we would like to temporally compress the tokens.

## References
[1] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

[2] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[3] Micheli, Vincent, Eloi Alonso, and François Fleuret. "Transformers are Sample-Efficient World Models." The Eleventh International Conference on Learning Representations. 2022.
