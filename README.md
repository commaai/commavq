# commavq
commaVQ is a dataset of 100,000 heavily compressed driving videos for Machine Learning research. A heavily compressed driving video like this is useful to experiment with GPT-like video prediction models. This repo includes an encoder/decoder and an example of a video prediction model.

## 2x$500 Challenges!

- Get 1.92 cross entropy loss or less in the val set and in our private val set (using `./notebooks/eval.ipynb`). gpt2m trained on a larger dataset gets 2.02 cross entropy loss.
- Make gpt2m.onnx run at 0.9 sec/frame or less on a consumer GPU (e.g. NVIDIA 3090) without degredation in cross entropy loss. The current implementation runs at 1.5 sec/frame with kvcaching and float16.

## Overview
A VQ-VAE [1,2] was used to heavily compress each frame into 128 "tokens" of 10 bits each. Each entry of the dataset is a "segment" of compressed driving video, i.e. 1min of frames at 20 FPS. Each file is of shape 1200x8x16 and saved as int16.

Note that the compressor is extremely lossy on purpose. It makes the dataset smaller and easy to play with (train GPT with large context size, fast autoregressive generation, etc.). We might extend the dataset to a less lossy version when we see fit.

## Download
- Using huggingface datasets
```python
import numpy as np
from datasets import load_dataset
num_proc = 40 # CPUs go brrrr
ds = load_dataset('commaai/commavq', num_proc=num_proc)
tokens = np.load(ds['0'][0]['path']) # first segment from the first data shard
```

- Manually download from huggingface datasets repository: https://huggingface.co/datasets/commaai/commavq

- From Academic Torrents (soon)


## Models
In ./models/ you will find 3 Neural Networks saved in the onnx format
- `./models/encoder.onnx`: is the encoder used to compress the frames
- `./models/decoder.onnx`: is the decoder used to decompress the frames
- `./models/gtp2m.onnx`: a 300M parameter GPT trained on a larger version of this dataset
- (experimental) `./models/temporal_decoder.onnx`: a temporal decoder which is a stateful version of the vanilla decoder

## Examples
Checkout `./nootebooks/encode.ipynb` and `./notebooks/decode.ipynb` for an example of how to visualize the dataset using a segment of driving video from [comma's drive to Taco Bell](https://blog.comma.ai/taco-bell/)

Checkout `./notebooks/gpt.ipynb` for an example of how to use a pretrained GPT model to imagine future frames.

https://github.com/commaai/commavq/assets/29985433/91894bf7-592b-4204-b3f2-3e805984045c

https://github.com/commaai/commavq/assets/29985433/3a799ac8-781e-461c-bf14-c15cea42b985

https://github.com/commaai/commavq/assets/29985433/f6f7699b-b6cb-4f9c-80c9-8e00d75fbfae


## References
[1] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

[2] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
