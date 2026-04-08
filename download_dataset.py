import numpy as np
from datasets import load_dataset
# load the first shard
data_files = {'train': ['data-0000.tar.gz']}
ds = load_dataset('commaai/commavq', data_files=data_files)
tokens = np.array(ds['train'][0]['token.npy'])
poses = np.array(ds['train'][0]['pose.npy'])