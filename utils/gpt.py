"""
adapted from https://github.com/pytorch-labs/gpt-fast
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

def find_multiple(n: int, k: int) -> int:
  if n % k == 0:
    return n
  return n + k - (n % k)

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
  q = torch.empty_like(probs_sort).exponential_(1)
  return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample(logits):
  probs = torch.nn.functional.softmax(logits[0, -1], dim=-1)
  idx_next = multinomial_sample_one_no_sync(probs)
  return idx_next, probs

@dataclass
class GPTConfig:
  block_size: int = 20*129
  vocab_size: int = 1025
  n_layer: int = 24
  n_head: int = 16
  dim: int = 1024
  intermediate_size: int = 4*1024
  tokens_per_frame: int = 129

  @property
  def bos_token(self):
    return self.vocab_size - 1

  @property
  def head_dim(self):
    return self.dim // self.n_head

class KVCache(nn.Module):
  def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
    super().__init__()
    cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
    self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
    self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

  def update(self, input_pos, k_val, v_val):
    assert input_pos.shape[0] == k_val.shape[2]
    k_out = self.k_cache
    v_out = self.v_cache
    k_out[:, :, input_pos] = k_val
    v_out[:, :, input_pos] = v_val
    return k_out, v_out

class TransformerBlock(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.attn = Attention(config)
    self.mlp = FeedForward(config)
    self.ln_1 = nn.LayerNorm(config.dim)
    self.ln_2 = nn.LayerNorm(config.dim)

  def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
    h = x + self.attn(self.ln_1(x), mask, input_pos)
    out = h + self.mlp(self.ln_2(h))
    return out

class Attention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.dim % config.n_head == 0
    self.config = config
    # key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.dim, 3*config.dim, bias=True)
    self.c_proj = nn.Linear(config.dim, config.dim, bias=True)

  def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
    bsz, seqlen, _ = x.shape

    q, k, v = self.c_attn(x).split([self.config.dim, self.config.dim, self.config.dim], dim=-1)

    q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
    k = k.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
    v = v.view(bsz, seqlen, self.config.n_head, self.config.head_dim)

    q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

    if self.kv_cache is not None:
      k, v = self.kv_cache.update(input_pos, k, v)

    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim)
    return self.c_proj(y)

class FeedForward(nn.Module):
  def __init__(self, config: GPTConfig) -> None:
    super().__init__()
    self.c_fc = nn.Linear(config.dim, config.intermediate_size, bias=True)
    self.c_proj = nn.Linear(config.intermediate_size, config.dim, bias=True)

  def forward(self, x: Tensor) -> Tensor:
    return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))

class GPT(nn.Module):
  def __init__(self, config: GPTConfig=GPTConfig()) -> None:
    super().__init__()
    self.config = config

    transformer = {
    'wte' : nn.Embedding(config.vocab_size, config.dim),
    'wpe' : nn.Embedding(config.block_size, config.dim),
    'h'   : nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer)),
    'ln_f' : nn.LayerNorm(config.dim)
    }

    self.transformer = nn.ModuleDict(transformer)
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    self.causal_mask: Optional[Tensor] = None
    self.max_batch_size = -1
    self.max_seq_length = -1

  def setup_caches(self, max_batch_size, max_seq_length):
    if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
      return
    max_seq_length = find_multiple(max_seq_length, 8)
    self.max_seq_length = max_seq_length
    self.max_batch_size = max_batch_size
    for b in self.transformer.h:
      b.attn.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, self.config.head_dim)

    self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).view(1, 1, self.max_seq_length, self.max_seq_length)

  def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
    mask = self.causal_mask[:, :, input_pos]
    x = self.transformer.wte(idx) + self.transformer.wpe(input_pos)

    for _, layer in enumerate(self.transformer.h):
      x = layer(x, input_pos, mask)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    return logits

  def prefill(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    logits = self(x, input_pos)
    return sample(logits)[0]

  def decode_one_token(self,  x: torch.Tensor, input_pos: torch.Tensor):
    assert input_pos.shape[-1] == 1
    logits = self(x, input_pos)
    return sample(logits)

  def decode_n_tokens(self, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int,):
    new_tokens, new_probs = [], []
    for _ in range(num_new_tokens):
      with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        next_token, next_prob = self.decode_one_token(cur_token, input_pos)
      input_pos += 1
      new_tokens.append(next_token.clone())
      new_probs.append(next_prob.clone())
      cur_token = next_token.view(1, -1)
    return new_tokens, new_probs

  @torch.no_grad()
  def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    t = prompt.size(0)
    T_new = t + max_new_tokens
    max_seq_length = self.config.block_size
    device, dtype = prompt.device, prompt.dtype

    with torch.device(device):
      self.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    seq = torch.empty(T_new, dtype=dtype, device=device).clone()
    seq[:t] = prompt
    input_pos = torch.arange(0, t, device=device)
    next_token = self.prefill(prompt.view(1, -1), input_pos).clone()
    seq[t] = next_token
    input_pos = torch.tensor([t], device=device, dtype=torch.int)
    generated_tokens, _ = self.decode_n_tokens(next_token.view(1, -1), input_pos, max_new_tokens - 1)
    seq[t+1:] = torch.cat(generated_tokens)
    return seq[t:]

  def load_state_dict_from_url(self,url='https://huggingface.co/commaai/commavq-gpt2m/resolve/main/pytorch_model.bin', *args, **kwargs):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', weights_only=True)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    state_dict = {k: v for k, v in state_dict.items() if not any([k.endswith('.attn.masked_bias'), k.endswith('.attn.bias')])}
    for k in state_dict.keys():
      if any(k.endswith(w) for w in transposed):
        state_dict[k] = torch.transpose(state_dict[k], 1, 0)
    self.load_state_dict(state_dict,  *args, **kwargs)
