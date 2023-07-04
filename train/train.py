import torch
import numpy as np

from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
from torch import optim

from model import Encoder, Decoder, Quantizer, N_FRAME_TOKS
from dataloader import TokenLoader


if  __name__ == "__main__":
    '''
    num_proc = 40 # CPUs go brrrr
    ds = load_dataset('commaai/commavq', num_proc=num_proc)


    tokens = np.load(ds['0'][0]['path']) # first segment from the first data shard
    '''
    # Input: [b, 2, N_FRAME_TOKS]
    # Flattened: [b, 256]

    # Bottleneck: [b, N_FRAME_TOKS + s] : s < 128 n Bottleneck[:128] = Input[:128]
    # I.e. Transformation_Code = Encoder(Input): [b, s]
    #      Output = Decoder(Input[:N_FRAME_TOKS] + Transformation_Code): [b, 128]

    # Output: [b, 2, N_FRAME_TOKS]

    '''
    For now try to get results with constant bottleneck
    i.e. try to get some encoder model that can encode the diffs in 50% less tokens

    Next try variable length codings based on difference between tokens of frames
    Need to add delimiter tokens <|X1|> abcd <|F|> 12 <|X2>| xzyw <|EOT|> pad pad pad 
    '''

    # Data Prep
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spatial_embeddings = torch.load("../embeddings.pt").to(device)
    spatial_embeddings.requires_grad = False

    batch_size = 2
    dataloader = TokenLoader('commavq-mini.npy', batch_size)

    # Model Prep

    N_DYNAMICS_TOKS = 64  # s = N_FRAME_TOKS - 64 = 128 / 2

    enc = Encoder(
        width=256,
        layers=2,
        heads=8,
        n_tokens=N_DYNAMICS_TOKS,
        spatial_embeddings=spatial_embeddings,
    ).to(device)
    dec = Decoder(
        width=256,
        layers=2,
        heads=8,
        n_tokens=N_DYNAMICS_TOKS,
        spatial_embeddings=spatial_embeddings,
    ).to(device)
    q = Quantizer(
        n_embeddings=256,
        embedding_dim=256,
        commitment_cost=0.25,
    ).to(device)

    # Opt Prep
    iters = 200

    loss_func = torch.nn.CrossEntropyLoss()
    opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()))

    i = 0
    for X in dataloader:
        if i >= iters:
            break
        X = X.long().to(device)

        embs = spatial_embeddings[X]
        e0, e1 = embs[:, :N_FRAME_TOKS], embs[:, -N_FRAME_TOKS:]
        labels = X[:, -N_FRAME_TOKS:]

        # Forward pass
        opt.zero_grad()

        f_emb = enc(X)

        f, ppl = q(f_emb)

        e0 = torch.zeros(e0.shape).to(device)

        logits = dec(e0, f, e1)
        true_logits = logits[:, -N_FRAME_TOKS:]

        prep_logits, prep_labels = true_logits.reshape(-1, 1024), labels.reshape(-1)
        reco_loss = loss_func(prep_logits, prep_labels)
        latent_loss = q.compute_latent_loss(f_emb, f)
        
        print(reco_loss, latent_loss)
        print(ppl)

        loss = reco_loss + latent_loss
        #2 loss = latent_loss

        loss.backward()
        opt.step()
        i += 1

    print(true_logits.argmax(dim=-1)[0])
    print(labels[0])
    print((true_logits.argmax(dim=-1)[0] == labels[0]).sum()/labels[0].numel())

