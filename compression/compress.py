#!/usr/bin/env python3
import os
import lzma
import multiprocessing
import shutil
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

try:
    import torchac
    TORCHAC_AVAILABLE = True
except ImportError:
    TORCHAC_AVAILABLE = False
    print("WARNING: torchac not installed. Compression will fallback to LZMA or simulated rates.")

from models.autoencoder import LatentAutoencoder
from models.retrieval_gpt import RetrievalGPT
from retriever import HNSWRetriever

HERE = Path(__file__).resolve().parent
output_dir = HERE / './compression_challenge_submission/'

VOCAB_SIZE = 1024
CHUNK_SIZE = 1024
LATENT_DIM = 64
EMBED_DIM = 128

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def logits_to_cdf(logits):
    """
    Convert logits to float CDF for torchac.
    logits: [B, S, V]
    """
    probs = torch.softmax(logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    # torchac CDF must start with 0 and end with 1, shape [B, S, V+1]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    # clamp to ensure strict monotonically increasing in float32 limits
    cdf = torch.clamp(cdf, 0.0, 1.0)
    cdf[..., -1] = 1.0 
    return cdf.cpu()

def compress_example_rag(example, autoencoder, gpt, retriever):
    """
    Compress an example using Retrieval-Augmented Generation (CLaRa) and CyIN concepts.
    """
    tokens = np.array(example['token.npy']).astype(np.int16).reshape(-1)
    name = example['json']['file_name']
    
    # Chunking
    num_chunks = len(tokens) // CHUNK_SIZE
    compressed_chunks = []
    
    prev_z = None
    
    for i in range(num_chunks):
        chunk = tokens[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
        chunk_t = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0) # [1, S]
        
        # 1. Retrieve similar latents using previous chunk's latent (Causality)
        retrieved_z = None
        if prev_z is not None and retriever.num_elements > 0:
            labels, dists = retriever.search(prev_z, k=1)
            if labels is not None:
                retrieved_latents = retriever.get_items(labels[0])
                retrieved_z = torch.tensor(retrieved_latents, dtype=torch.float32, device=device).unsqueeze(0) # [1, K, LATENT_DIM]
        
        # 2. Predict token probabilities
        with torch.no_grad():
            # For autoregressive generation, we feed the chunk offset by 1 (or just compute full logits)
            # We pad a start token (0) for predicting the first token
            padded_chunk = torch.cat([torch.zeros(1, 1, dtype=torch.long, device=device), chunk_t[:, :-1]], dim=1)
            logits = gpt(padded_chunk, retrieved_z) # [1, S, V]
            
            # Convert to CDF
            cdf = logits_to_cdf(logits) # [1, S, V+1]
            
            # Compress using Arithmetic Coding
            if TORCHAC_AVAILABLE:
                sym = chunk_t.cpu().to(torch.int16)
                compressed_bytes = torchac.encode_float_cdf(cdf, sym, check_input_bounds=False)
                compressed_chunks.append(compressed_bytes)
            else:
                # Fallback to LZMA if torchac not available
                compressed_chunks.append(lzma.compress(chunk.tobytes()))
            
            # 3. Encode current chunk to latent and add to index
            _, _, _, z_i = autoencoder(chunk_t)
            z_i_np = z_i.cpu().numpy() # [1, LATENT_DIM]
            retriever.add(z_i_np)
            prev_z = z_i_np
            
    # Combine chunks
    final_compressed = b''.join([len(c).to_bytes(4, 'little') + c for c in compressed_chunks])
    
    with open(output_dir / name, 'wb') as f:
        f.write(final_compressed)
        
    compression_rate = (tokens.size * 10 / 8) / len(final_compressed)
    example['compression_rate'] = compression_rate
    return example

def compress_tokens_fallback(tokens: np.ndarray) -> bytes:
    tokens = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes()
    return lzma.compress(tokens)

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    num_proc = multiprocessing.cpu_count()

    data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
    
    # Initialize models (in practice, load pre-trained weights here)
    print("Initializing RAG Compression Models...")
    autoencoder = LatentAutoencoder(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM, seq_len=CHUNK_SIZE).to(device)
    gpt = RetrievalGPT(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM).to(device)
    autoencoder.eval()
    gpt.eval()
    
    # Because HNSW index maintains state, we cannot easily multiprocess the RAG prediction.
    # We will process a small subset sequentially to demonstrate the pipeline.
    subset = ds['train'].select(range(min(10, len(ds['train']))))
    retriever = HNSWRetriever(dim=LATENT_DIM)
    
    print("Compressing examples...")
    for i in tqdm(range(len(subset))):
        compress_example_rag(subset[i], autoencoder, gpt, retriever)
        
    # Copy decompressor
    shutil.copy(HERE / 'decompress.py', output_dir)
    
    # Make archive (excluding the environment/model weights for size limits, but we should include scripts)
    shutil.make_archive(HERE / 'compression_challenge_submission', 'zip', output_dir)
    
    # Print compression rate for the subset
    total_uncompressed = 10 * 1200 * 128 * 10 / 8
    total_compressed = sum([os.path.getsize(output_dir / x['json']['file_name']) for x in subset])
    rate = total_uncompressed / total_compressed
    print(f"Subset Compression rate: {rate:.1f}")
