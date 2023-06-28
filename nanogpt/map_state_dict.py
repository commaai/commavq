"""
map_state_dict is a helper function to map the state dict from gpt2m.pt to a nanogpt model
use this to finetune using the nanogpt training code!

# usage:
import torch
from model import GPT, GPTConfig

config = GPTConfig(
    block_size=(128+1)*20, # 20 frames 128 tokens per frame, plus 1 for BOS_TOKEN
    vocab_size=1025,
    n_layer=24,
    n_head=16,
    n_embd=1024,
    dropout=0.0,
    bias=True,
)

model = GPT(config)

# load the state dict
state_dict = torch.load('models/gpt2m.pt', map_location=torch.device('cpu'))['state_dict']
state_dict = map_state_dict(state_dict)
model.load_state_dict(state_dict)
"""


def map_state_dict(state_dict):
  key_map = {
    'transformer.wt_embedding.weight'       : 'transformer.wte.weight',
    'transformer.wp_embedding.weight'       : 'transformer.wpe.weight',
    **{f'transformer.h.{i}.attn.layer_norm.weight': f'transformer.h.{i}.ln_1.weight' for i in range(24)},
    **{f'transformer.h.{i}.attn.layer_norm.bias'  : f'transformer.h.{i}.ln_1.bias' for i in range(24)},
    **{f'transformer.h.{i}.mlp.layer_norm.weight': f'transformer.h.{i}.ln_2.weight' for i in range(24)},
    **{f'transformer.h.{i}.mlp.layer_norm.bias': f'transformer.h.{i}.ln_2.bias' for i in range(24)},
    'transformer.layer_norm_f.weight': 'transformer.ln_f.weight',
    'transformer.layer_norm_f.bias': 'transformer.ln_f.bias',
  }

  state_dict = {key_map.get(k, k): v for k, v in state_dict.items()}
  return state_dict
