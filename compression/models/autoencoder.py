import torch
import torch.nn as nn

class LatentAutoencoder(nn.Module):
    """
    Autoencoder for compressing discrete video tokens into a continuous latent space.
    Uses Parameter-Efficient pooling to avoid bloating decompression payload weights.
    """
    def __init__(self, vocab_size=1024, embed_dim=256, latent_dim=64, seq_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Simple Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # IB Parameters (applied after average pooling)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        # Decoder
        self.decoder_proj = nn.Linear(latent_dim, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=1024, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        
        self.out_head = nn.Linear(embed_dim, vocab_size)
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
    def encode(self, x):
        # x: [B, S]
        emb = self.embed(x)
        hidden = self.encoder(emb)
        # Parameter-efficient pooling over sequence
        pooled = hidden.mean(dim=1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        # z: [B, latent_dim]
        z_proj = self.decoder_proj(z) # [B, embed_dim]
        z_seq = z_proj.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        pos = torch.arange(0, self.seq_len, dtype=torch.long, device=z.device).unsqueeze(0)
        z_seq = z_seq + self.pos_emb(pos)
        
        decoded = self.decoder(z_seq)
        logits = self.out_head(decoded)
        return logits
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z
