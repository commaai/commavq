
| Source Video    | Compressed Video | Future Prediction |
| --------------- | ---------------- |------------------ |
| <video src="https://github.com/commaai/commavq/assets/29985433/91894bf7-592b-4204-b3f2-3e805984045c">  |  <video src="https://github.com/commaai/commavq/assets/29985433/3a799ac8-781e-461c-bf14-c15cea42b985">    |  <video src="https://github.com/commaai/commavq/assets/29985433/f6f7699b-b6cb-4f9c-80c9-8e00d75fbfae"> |

A world model is a model that can predict the next state of the world given the observed previous states and actions.

World models are essential to training all kinds of intelligent agents, especially self-driving models.

commaVQ contains:
- encoder/decoder models used to heavily compress driving scenes
- a world model trained on 3,000,000 minutes of driving videos
- a dataset of 100,000 minutes of compressed driving videos

# Tasks

## Lossless compression challenge: make me smaller! $500 challenge
Losslessly compress 5,000 minutes of driving video. Go to [./compression/](./compression/) to start

**Prize: highest compression rate on 5,000 minutes of driving video (~915MB) - Challenge ends July, 1st 2024 11:59pm AOE**

Note: submit a single zip file containing the compressed data and a python script to decompress it into its original form. Everything in this repository and in PyPI is assumed to be available.

| Implementation                                                                     | Compression rate |
| :----------------------------------------------------------------------------------| ---------------: |
| lzma                                                                               |  1.6             |


## Overview
A VQ-VAE [1,2] was used to heavily compress each video frame into 128 "tokens" of 10 bits each. Each entry of the dataset is a "segment" of compressed driving video, i.e. 1min of frames at 20 FPS. Each file is of shape 1200x8x16 and saved as int16.

A world model [3] was trained to predict the next token given a context of past tokens. This world model is a Generative Pre-trained Transformer (GPT) [4] trained on 3,000,000 minutes of driving videos following a similar recipe to [5].

## Examples
[./notebooks/encode.ipynb](./notebooks/encode.ipynb) and [./notebooks/decode.ipynb](./notebooks/decode.ipynb) for an example of how to visualize the dataset using a segment of driving video from [comma's drive to Taco Bell](https://blog.comma.ai/taco-bell/)

[./notebooks/gpt.ipynb](./notebooks/gpt.ipynb) for an example of how to use the world model to imagine future frames.

[./notebooks/compress.ipynb](./notebooks/compress.ipynb) for an example of how we would like to temporally compress the tokens of each segment

## References
[1] Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

[2] Esser, Patrick, Robin Rombach, and Bjorn Ommer. "Taming transformers for high-resolution image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

[3] https://worldmodels.github.io/

[4] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[5] Micheli, Vincent, Eloi Alonso, and François Fleuret. "Transformers are Sample-Efficient World Models." The Eleventh International Conference on Learning Representations. 2022.
