import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

#load_dataset('cerebras/SlimPajama-627B')
u = load_dataset('venketh/SlimPajama-62B')

print('finished')
from IPython import embed; embed()