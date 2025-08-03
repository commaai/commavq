#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import multiprocessing
from datasets import load_dataset, DatasetDict

archive_path = Path(os.environ.get('PACKED_ARCHIVE', './compression_challenge_submission.zip'))
unpacked_archive = Path(os.environ.get('UNPACKED_ARCHIVE', './compression_challenge_submission_decompressed/'))

def compare(example):
  name = example['json']['file_name']
  tokens = np.load(unpacked_archive/name)
  gt_tokens = example['token.npy']
  assert np.all(tokens == gt_tokens), f"decompressed data does not match original data for {path}"

if __name__ == '__main__':
  num_proc = multiprocessing.cpu_count()
  # load split 0 and 1
  data_files = {'train': ['data-0000.tar.gz', 'data-0001.tar.gz']}
  ds = load_dataset('commaai/commavq', num_proc=num_proc, data_files=data_files)
  # compare
  ds.map(compare, desc="compare", num_proc=num_proc, load_from_cache_file=False)
  # print compression rate
  rate = (sum(ds.num_rows.values()) * 1200 * 128 * 10 / 8) / archive_path.stat().st_size
  print(f"Compression rate: {rate:.1f}")
