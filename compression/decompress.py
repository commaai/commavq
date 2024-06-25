#!/usr/bin/env python3
import os
import lzma
import numpy as np
from pathlib import Path
import multiprocessing
from datasets import load_dataset, DatasetDict

HERE = Path(__file__).resolve().parent

output_dir = Path(os.environ.get('OUTPUT_DIR', HERE/'./compression_challenge_submission_decompressed/'))

def decompress_bytes(x: bytes) -> np.ndarray:
  tokens = np.frombuffer(lzma.decompress(x), dtype=np.int16)
  return tokens.reshape(128, -1).T.reshape(-1, 8, 16)

def decompress_example(example):
  path = Path(example['path'])
  with open(output_dir/path.name, 'rb') as f:
    tokens = decompress_bytes(f.read())
  np.save(output_dir/path.name, tokens)
  assert np.all(tokens == np.load(path)), f"decompressed data does not match original data for {path}"

if __name__ == '__main__':
  num_proc = multiprocessing.cpu_count()
  # load split 0 and 1
  splits = ['0', '1']
  ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits)
  ds = DatasetDict(zip(splits, ds))
  # decompress
  ds.map(decompress_example, desc="decompress_example", num_proc=num_proc, load_from_cache_file=False)
