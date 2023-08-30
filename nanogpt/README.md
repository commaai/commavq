# nanoGPT helpers

`./nanogpt/prepare.py` is provided to convert the dataset to the format used in [nanoGPT](https://github.com/karpathy/nanoGPT)

Use the huggingface transformer library to download the model checkpoint, or load it in a `GPT2LMHeadModel` from `./gpt2m/pytorch_model.bin`

```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("commaai/commavq-gpt2m")
```
