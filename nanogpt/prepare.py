# saves commaVQ in a format that can be used by nanogpt.train.py
# this writes 40 files of 774MB each, for a total of 30GB
# modified from https://github.com/karpathy/nanoGPT/tree/master/data
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 40
num_proc_load_dataset = num_proc

# because nanoGPT datasets are flat arrays of tokens
# we use this token to separate frames
BOS_TOKEN = 1024
# we use this token to separate segments
# note that the gpt2m is only trained on tokens from the same segment and doesn't have an EOT_TOKEN
EOT_TOKEN = 1025

if __name__ == '__main__':

  # takes 17GB in huggingface .cache dir, 100k segments
  dataset = load_dataset("commaai/commavq", num_proc=num_proc_load_dataset)

  def process(example):
    tokens = np.load(example['path'])
    tokens = tokens.reshape(tokens.shape[0], -1)
    # prepend BOS_TOKEN
    tokens = np.c_[np.ones(len(tokens), dtype=np.int16)*BOS_TOKEN, tokens]
    tokens = tokens.reshape(-1)
    # append EOT_TOKEN
    tokens = np.r_[tokens, EOT_TOKEN]
    return {'ids': tokens.astype(np.int16), 'len': len(tokens.astype(np.int16))}

  # the dataset is already tokenized, but need to ad BOS and EOT tokens and flatten
  # this will be cached
  tokenized = dataset.map(
    process,
    desc="tokenizing the splits",
    num_proc=num_proc,
  )

  # concatenate all the ids in each split into one large file we can use for training
  for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since max_token_value == 1025 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 100 if split == '40' else 1024 # last split is the val set and is smaller
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
      # Batch together samples for faster write
      batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
      arr_batch = np.concatenate(batch['ids'])
      # Write into mmap
      arr[idx : idx + len(arr_batch)] = arr_batch
      idx += len(arr_batch)
    arr.flush()
