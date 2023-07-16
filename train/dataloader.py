import time
import torch
import numpy as np

from datasets import load_dataset


class TokenLoader:
    def __init__(self, ds_loc, batch_size, n_frames=2):
        self.batch_size = batch_size
        self.n_frames = n_frames

        self.ds = np.load(ds_loc)

    def __iter__(self):
        while True:
            shard_sample = np.random.randint(0, self.ds.shape[0], self.batch_size)
            segment_sample = np.random.randint(0, self.ds.shape[1], self.batch_size)
            frame_sample = np.random.randint(0, self.ds.shape[2] - self.n_frames + 1, self.batch_size)

            slices = []
            for i in range(self.n_frames):
                slices.append(self.ds[shard_sample, segment_sample, frame_sample+i].reshape(self.batch_size, 1, 8, 16))

            batch = np.concatenate(slices, axis=1)
            yield torch.from_numpy(batch)


if __name__ == "__main__":
    # Quick benchmark
    iters = 100
    batch_size = 32
    n_frames = 5

    tl = TokenLoader('commavq-mini.npy', batch_size, n_frames)

    t0 = time.time()
    i = 0
    for b in tl:
        if i >= iters:
            break
        print(b.shape)
        i += 1
    tf = time.time()

    print("tokens/s:")
    print(iters*batch_size*n_frames*8*16/(tf-t0))
