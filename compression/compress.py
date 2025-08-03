#!/usr/bin/env python3
import os
import lzma
import multiprocessing
import shutil
import numpy as np

from pathlib import Path
from datasets import load_dataset, DatasetDict

HERE = Path(__file__).resolve().parent

output_dir = HERE / './compression_challenge_submission/'

def compress_tokens(tokens: np.ndarray) -> bytes:
  tokens = tokens.astype(np.int16).reshape(-1, 128).T.ravel().tobytes() # transposing increases compression rate ;)
  return lzma.compress(tokens)

def compress_example(example):
  tokens = np.array(example['token.npy'])
  name = example['json']['file_name'] # or example['__key__']
  compressed = compress_tokens(tokens)
  compression_rate = (tokens.size * 10 / 8) / len(compressed) # 10 bits per token
  with open(output_dir/name, 'wb') as f:
    f.write(compressed)
  example['compression_rate'] = compression_rate
  return example

if __name__ == '__main__':
  os.makedirs(output_dir, exist_ok=True)
  num_proc = multiprocessing.cpu_count()

  # load split 0 and 1
  data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
  ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)

  # compress
  ratios = ds.map(compress_example, desc="compress_example", num_proc=num_proc, load_from_cache_file=False)

  # make archive
  shutil.copy(HERE/'decompress.py', output_dir)
  shutil.make_archive(HERE/'compression_challenge_submission', 'zip', output_dir)

  # print compression rate
  rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / os.path.getsize(HERE/"compression_challenge_submission.zip")
  print(f"Compression rate: {rate:.1f}")
