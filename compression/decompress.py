#!/usr/bin/python
"""
the decompression file should be in your submission archive
your archive can be packed in any format, as long as you send us a script to unpack it (e.g. unpack_archive.sh)
this script should save the decompressed files back to their original format
we will run compression/evaluate.py to compare the decompressed files with the original files and confirm the compression rate
"""
import lzma
import numpy as np
from pathlib import Path
import multiprocessing
from datasets import load_dataset, DatasetDict

output_dir = Path('./compression_challenge_submission_decompressed/')

def decompress_bytes(bytes: bytes) -> np.ndarray:
  tokens = np.frombuffer(lzma.decompress(bytes), dtype=np.int16)
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
