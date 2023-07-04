import time
import torch
import numpy as np

from datasets import load_dataset


class TokenLoader:
    def __init__(self, ds_loc, batch_size):
        '''
        Args:
            cache_size - how many videos to pre-load
            num_proc - speed up downloading (if not downloaded)
        '''
        self.batch_size = batch_size

        self.ds = np.load(ds_loc)

    def __iter__(self):
        while True:
            shard_sample = np.random.randint(0, self.ds.shape[0], self.batch_size).reshape(1, self.batch_size)
            segment_sample = np.random.randint(0, self.ds.shape[1], self.batch_size).reshape(1, self.batch_size)
            frame_sample = np.random.randint(0, self.ds.shape[2] - 1, self.batch_size).reshape(1, self.batch_size)

            ind = np.stack([shard_sample, segment_sample, frame_sample], axis=-1)[0]
            batch_0 = self.ds[ind[:, 0], ind[:, 1], ind[:, 2]].reshape(self.batch_size, -1)
            batch_1 = self.ds[ind[:, 0], ind[:, 1], ind[:, 2] + 1].reshape(self.batch_size, -1)

            batch = np.concatenate((batch_0, batch_1), axis=-1)
            yield torch.from_numpy(batch)


if __name__ == "__main__":
    # Quick benchmark
    iters = 100
    batch_size = 32

    tl = TokenLoader('commavq-mini.npy', batch_size)

    t0 = time.time()
    i = 0
    for b in tl:
        if i >= iters:
            break
        print(b.shape)
        i += 1
    tf = time.time()

    print(iters*batch_size/(tf-t0))
