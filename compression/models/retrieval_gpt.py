import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class RetrievalGPT(nn.Module):
    """
    Predictive GPT model that implements Retrieval-Augmented Generation (CLaRa).
    Takes a sequence of tokens and prepended retrieved latents (as context).
    Scaled to 10-20M parameters to minimize decompression payload size.
    """
    def __init__(self, vocab_size=1024, embed_dim=256, num_layers=4, num_heads=8, latent_dim=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim)
        
        # Project retrieved latent back into the embedding space
        self.latent_proj = nn.Linear(latent_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x, retrieved_z=None):
        # x: [B, S]
        emb = self.token_emb(x)
        emb = self.pos_emb(emb)
        
        if retrieved_z is not None:
            # retrieved_z: [B, K, latent_dim]
            z_emb = self.latent_proj(retrieved_z) # [B, K, embed_dim]
            # Concat as prefix context
            emb = torch.cat([z_emb, emb], dim=1)
            
        # Create causal mask
        total_len = emb.size(1)
        # Upper triangular mask filled with -inf
        mask = torch.triu(torch.ones(total_len, total_len) * float('-inf'), diagonal=1).to(x.device)
        
        out = self.transformer(emb, mask=mask, is_causal=True)
        
        # Extract outputs corresponding to x
        if retrieved_z is not None:
            K = retrieved_z.size(1)
            out = out[:, K:, :]
            
        logits = self.lm_head(out)
        return logits
