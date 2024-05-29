"""
adapted from: https://github.com/CompVis/taming-transformers
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from einops import rearrange

@dataclass
class CompressorConfig:
  in_channels: int = 3
  out_channels: int = 3
  ch_mult: tuple[int] = (1,1,2,2,4)
  attn_resolutions: tuple[int] = (16,)
  resolution: int = 256
  num_res_blocks: int = 2
  z_channels: int = 256
  vocab_size: int = 1024
  ch: int = 128
  dropout: float = 0.0

  @property
  def num_resolutions(self):
    return len(self.ch_mult)

  @property
  def quantized_resolution(self):
    return self.resolution // 2**(self.num_resolutions-1)

def nonlinearity(x): # swish
  return x*torch.sigmoid(x)

def Normalize(in_channels):
  return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
  def forward(self, x):
    x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    return self.conv(x)

class Downsample(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    # no asymmetric padding in torch conv, must do it ourselves
    self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

  def forward(self, x):
    pad = (0,1,0,1)
    x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
    return self.conv(x)


class ResnetBlock(nn.Module):
  def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
         dropout, temb_channels=512):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels
    self.use_conv_shortcut = conv_shortcut

    self.norm1 = Normalize(in_channels)
    self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    if temb_channels > 0:
      self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
    self.norm2 = Normalize(out_channels)
    self.dropout = torch.nn.Dropout(dropout)
    self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
      else:
        self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x, temb):
    h = x
    h = self.norm1(h)
    h = nonlinearity(h)
    h = self.conv1(h)

    if temb is not None:
      h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

    h = self.norm2(h)
    h = nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        x = self.conv_shortcut(x)
      else:
        x = self.nin_shortcut(x)

    return x+h

class AttnBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.norm = Normalize(in_channels)
    self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q = q.reshape(b,c,h*w)
    q = q.permute(0,2,1)    # b,hw,c
    k = k.reshape(b,c,h*w)  # b,c,hw
    w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b,c,h*w)
    w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
    h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = h_.reshape(b,c,h,w)

    h_ = self.proj_out(h_)
    return x+h_

class VectorQuantizer(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings

    self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
    self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)

  # the encode function
  def forward(self, inputs):
    b, s, c = inputs.shape
    flat_input = rearrange(inputs, 'b s c -> (b s) c')

    # Calculate distances
    distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                + torch.sum(self._embedding.weight**2, dim=1)
                - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

    # Encoding
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    quantized = self.embed(encoding_indices)
    quantized = rearrange(quantized, '(b s) c -> b s c', b=b, s=s, c=c).contiguous()
    encoding_indices = rearrange(encoding_indices, '(b s) 1 -> b s', b=b, s=s)
    return quantized, encoding_indices

  # the decode function
  def decode(self, encoding_indices):
    b, s = encoding_indices.shape
    encoding_indices = rearrange(encoding_indices, 'b s -> (b s) 1', b=b, s=s)
    quantized = self.embed(encoding_indices)
    quantized = rearrange(quantized, '(b s) c -> b s c', b=b, c=self._embedding_dim, s=s).contiguous()
    encoding_indices = rearrange(encoding_indices, '(b s) 1 -> b s', b=b, s=s)
    return quantized, encoding_indices

  def embed(self, encoding_indices):
    encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
    encodings.scatter_(1, encoding_indices, 1)
    quantized = torch.matmul(encodings, self._embedding.weight)
    return quantized

class Encoder(nn.Module):
  def __init__(self, config: CompressorConfig):
    super().__init__()
    self.config = config
    self.temb_ch = 0
    # downsampling
    self.conv_in = torch.nn.Conv2d(self.config.in_channels, self.config.ch, kernel_size=3, stride=1, padding=1)

    curr_res = self.config.resolution
    in_ch_mult = (1,)+tuple(self.config.ch_mult)
    self.down = nn.ModuleList()
    for i_level in range(self.config.num_resolutions):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_in = self.config.ch*in_ch_mult[i_level]
      block_out = self.config.ch*self.config.ch_mult[i_level]
      for _ in range(self.config.num_res_blocks):
        block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=self.config.dropout))
        block_in = block_out
        if curr_res in self.config.attn_resolutions:
          attn.append(AttnBlock(block_in))
      down = nn.Module()
      down.block = block
      down.attn = attn
      if i_level != self.config.num_resolutions-1:
        down.downsample = Downsample(block_in)
        curr_res = curr_res // 2
      self.down.append(down)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout)
    # end
    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(block_in, self.config.z_channels, kernel_size=3, stride=1, padding=1)

    # quantizer
    self.quant_conv = torch.nn.Conv2d(self.config.z_channels, self.config.z_channels, 1)
    self.quantize = VectorQuantizer(self.config.vocab_size, self.config.z_channels)


  def forward(self, x):
    # timestep embedding
    temb = None

    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.config.num_resolutions):
      for i_block in range(self.config.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1], temb)
        if len(self.down[i_level].attn) > 0:
          h = self.down[i_level].attn[i_block](h)
        hs.append(h)
      if i_level != self.config.num_resolutions-1:
        hs.append(self.down[i_level].downsample(hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # end
    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)

    # run the encoder part of VQ
    h = self.quant_conv(h)
    h = rearrange(h, 'b c h w -> b (h w) c')
    _, encoding_indices = self.quantize(h)
    return encoding_indices

  def load_state_dict_from_url(self, url='https://huggingface.co/commaai/commavq-gpt2m/resolve/main/encoder_pytorch_model.bin', *args, **kwargs):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', weights_only=True)
    self.load_state_dict(state_dict, *args, **kwargs)

class Decoder(nn.Module):
  def __init__(self, config: CompressorConfig):
    super().__init__()
    self.temb_ch = 0
    self.config = config

    # compute in_ch_mult, block_in and curr_res at lowest res
    block_in = self.config.ch*self.config.ch_mult[self.config.num_resolutions-1]
    curr_res = self.config.quantized_resolution

    # quantizer
    self.post_quant_conv = torch.nn.Conv2d(config.z_channels, config.z_channels, 1)
    self.quantize = VectorQuantizer(config.vocab_size, config.z_channels)

    # z to block_in
    self.conv_in = torch.nn.Conv2d(self.config.z_channels, block_in, kernel_size=3, stride=1, padding=1)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=self.config.dropout)

    # upsampling
    self.up = nn.ModuleList()
    for i_level in reversed(range(self.config.num_resolutions)):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_out = self.config.ch*self.config.ch_mult[i_level]
      for _ in range(self.config.num_res_blocks+1):
        block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=self.config.dropout))
        block_in = block_out
        if curr_res in self.config.attn_resolutions:
          attn.append(AttnBlock(block_in))
      up = nn.Module()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in)
        curr_res = curr_res * 2
      self.up.insert(0, up) # prepend to get consistent order

    # end
    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(block_in, self.config.out_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, encoding_indices):
    # run the decoder part of VQ
    z, _ = self.quantize.decode(encoding_indices)
    z = rearrange(z, 'b (h w) c -> b c h w', w=self.config.quantized_resolution)
    z = self.post_quant_conv(z)
    self.last_z_shape = z.shape

    # timestep embedding
    temb = None

    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # upsampling
    for i_level in reversed(range(self.config.num_resolutions)):
      for i_block in range(self.config.num_res_blocks+1):
        h = self.up[i_level].block[i_block](h, temb)
        if len(self.up[i_level].attn) > 0:
          h = self.up[i_level].attn[i_block](h)
      if i_level != 0:
        h = self.up[i_level].upsample(h)

    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)

    # scale
    return ((h + 1.0) / 2.0) * 255.

  def load_state_dict_from_url(self, url='https://huggingface.co/commaai/commavq-gpt2m/resolve/main/decoder_pytorch_model.bin', *args, **kwargs):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', weights_only=True)
    self.load_state_dict(state_dict,  *args, **kwargs)
