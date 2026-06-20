import torch
import torch.nn as nn

class LatentAutoencoder(nn.Module):
    """
    Autoencoder for compressing discrete video tokens into a continuous latent space.
    Implements Information Bottleneck (IB) principles to extract task-relevant features.
    """
    def __init__(self, vocab_size=1024, embed_dim=128, latent_dim=64, seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # Information Bottleneck parameters
        self.fc_mu = nn.Linear(embed_dim * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim * seq_len, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, embed_dim * seq_len)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size)
        )
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
    def encode(self, x):
        # x: [B, S]
        emb = self.embed(x) # [B, S, E]
        hidden = self.encoder(emb)
        hidden_flat = hidden.view(x.size(0), -1)
        mu = self.fc_mu(hidden_flat)
        logvar = self.fc_logvar(hidden_flat)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        # z: [B, L]
        hidden_flat = self.decoder_input(z)
        hidden = hidden_flat.view(-1, self.seq_len, self.embed_dim)
        logits = self.decoder(hidden)
        return logits
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z
