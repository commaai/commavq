import torch
import time
import numpy as np
import wandb

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


    Current Objective:
        Get the model to learn the spatial embedding table by passing it through the dynamics
        bottleneck

        N_DYN_TOKS == N_SPATIAL_TOKS

    '''

    # Logging
    enable_wandb = True

    if enable_wandb:
        wandb.init(
            project="diff-encoding",
        )

    # Data Prep
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spatial_embeddings = torch.load("embedding.pt").to(device)
    spatial_embeddings.requires_grad = False

    batch_size = 1
    n_frames = 2
    dataloader = TokenLoader('commavq-mini.npy', batch_size, n_frames=n_frames)

    # Model Prep
    N_DYNAMICS_TOKS = 64 # s = N_FRAME_TOKS - 64 = 128 / 2

    enc = Encoder(
        width=256,
        layers=8,
        heads=8,
        n_tokens=N_DYNAMICS_TOKS,
        n_input_tokens=n_frames*128 + n_frames,
        spatial_embeddings=spatial_embeddings,
    ).to(device)
    dec = Decoder(
        width=256,
        layers=8,
        heads=8,
        n_tokens=N_DYNAMICS_TOKS,
        # n_input_tokens=N_DYNAMICS_TOKS + N_FRAME_TOKS + 2,
        n_input_tokens=N_FRAME_TOKS,
        spatial_embeddings=spatial_embeddings,
    ).to(device)
    '''
    q = Quantizer(
        n_embeddings=128,
        embedding_dim=256,
        commitment_cost=0.25,
    ).to(device)
    '''

    # Opt Prep
    iters = 10000000

    # opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()) + list(q.parameters()))
    opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()))

    i = 0
    t0 = time.time()
    for X in dataloader:
        if i >= iters:
            break
        X = X.long().to(device)

        data_time = time.time() - t0

        embs = spatial_embeddings[X].reshape(X.shape[0], X.shape[1], -1, spatial_embeddings.shape[-1])
        e0 = embs[:, 0]
        X0 = X[:, 0].reshape(X.shape[0], -1).long()
        labels = X[:, 1:].reshape(X.shape[0], -1)

        # Forward pass
        opt.zero_grad()

        f_emb = enc(X)

        # f, ppl, encodings = q(f_emb)
        f = f_emb

        # logits = dec(e0, f)
        logits = dec(X0, f)

        # TODO: why does this matter???
        true_logits = logits[:, :N_FRAME_TOKS]
        # true_logits = logits[:, -N_FRAME_TOKS:]

        prep_logits, prep_labels = true_logits.reshape(-1, 1024), labels.reshape(-1)
        reco_loss = F.cross_entropy(prep_logits, prep_labels)
        # latent_loss = q.compute_latent_loss(f_emb, f)
        
        # loss = reco_loss + latent_loss
        # loss = latent_loss
        loss = reco_loss

        loss.backward()
        opt.step()

        batch_time = time.time() - t0

        log = {
            "perf/step": i,
            "perf/data_time": data_time,
            "perf/batch_time": batch_time,
        }

        # Check if you're using f embedding
        with torch.no_grad():
            fake_f = torch.randn(f.shape).to(f.device)
            # fake_logits = dec(e0, fake_f)
            fake_logits = dec(X0, fake_f)

            fake_logits = fake_logits[:, :N_FRAME_TOKS]
            fake_prep_logits = fake_logits.reshape(-1, 1024)
            unused_f_loss = F.cross_entropy(fake_prep_logits, prep_labels)
            log['train/unused_f_loss'] = unused_f_loss.item()
    

        pred = true_logits.argmax(dim=-1)
        x0 = X0
        x1 = labels

        pred_x0_acc = (pred == x0).sum()/x0.numel()
        pred_x1_acc = (pred == x1).sum()/x1.numel()
        x0_x1_eq = (x0 == x1).sum()/x1.numel()

        log["train/reco_loss"] = reco_loss.item()
        log["train/pred_x0_acc"] = pred_x0_acc.item()
        log["train/pred_x1_acc"] = pred_x1_acc.item()
        log["train/x0_x1_eq"] = x0_x1_eq.item()

        print(f"Step {i}")
        print("--------")
        for name, val in log.items():
            print(f"{name}: {val}")
        if enable_wandb:
            wandb.log(log)

        i += 1
        t0 = time.time()

    last_pred = pred[0]
    last_x0 = x0[0]
    last_x1 = x1[0]

    print('====== X0')
    print(last_x0)
    print('====== X1')
    print(last_x1)
    print('====== PRED')
    print(last_pred)
    print('======')
    print("pred - x0")
    print((last_pred == last_x0).sum()/last_x0.numel())
    print("pred - x1")
    print((last_pred == last_x1).sum()/last_x1.numel())
    print("x0 - x1")
    print((last_x0 == last_x1).sum()/last_x1.numel())

