import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from transformer import Transformer


N_FRAME_TOKS = 128


class Encoder(nn.Module):
    def __init__(self, width, layers, heads, n_tokens, spatial_embeddings):
        super().__init__()
        self.transformer = Transformer(width, layers, heads)
        self.n_tokens = n_tokens
        self.register_buffer('spatial_embeddings', spatial_embeddings, persistent=False)
    def forward(self, x):
        embs = self.spatial_embeddings[x]
        c_embs = self.transformer(embs)
        f = c_embs[:, :self.n_tokens]  # transformation is bottlenecked
        return f


class Decoder(nn.Module):
    def __init__(self, width, layers, heads, n_tokens, spatial_embeddings):
        super().__init__()
        self.transformer = Transformer(width, layers, heads)
        # TODO: weight tying here? 
        self.pred_head = nn.Linear(width, 1024, bias=False)

        full_attn_mask = self.build_attention_mask(2 * N_FRAME_TOKS + n_tokens)
        self.register_buffer('attn_mask', full_attn_mask, persistent=False)
        self.register_buffer('spatial_embeddings', spatial_embeddings, persistent=False)

    def build_attention_mask(self, ctx_len):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf

        # TODO: might need to build attn mask every time for dynamic stuff
        mask = torch.empty(ctx_len, ctx_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
        
    def forward(self, x, f, y):
        # x: [b, 128, 256]; f: [b, s, 256]
        fx = torch.cat([x, f, y], dim=1)  # concat space code with transformation code

        fx = fx.permute(1, 0, 2)  # NLD -> LNDsq

        y = self.transformer(fx, attn_mask=self.attn_mask)
        # y = y[:, :x.shape[1]]

        y = y.permute(1, 0, 2)  # LND -> NLD
        logits = self.pred_head(y)
        return logits


class Quantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/n_embeddings, 1/self.n_embeddings)

    def compute_latent_loss(self, f_emb, quantized):
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), f_emb)
        q_latent_loss = F.mse_loss(quantized, f_emb.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        return loss

    def forward(self, f_emb):
        flat_input = f_emb.reshape(-1, self.embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embeddings, device=f_emb.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(f_emb.shape)
        
        quantized = f_emb + (quantized - f_emb).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, perplexity
