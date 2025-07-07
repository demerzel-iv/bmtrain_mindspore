import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tqdm import tqdm
from typing import Iterable
from mindspore import Tensor, ops
from datasets import load_dataset
from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer

import mindspore as ms
import bmtrain_mindspore as bms
import matplotlib.pyplot as plt

from data_utils import DistributedIteratorDataLoader

TOKENIZER_PATH = '/root/thunlp/data/DeepSeek-V2-Lite-ms'
DATASET_NAME = 'venketh/SlimPajama-62B'

def iter_func(text_items) -> Iterable[str]:
    cnt = 0
    for item in text_items:
        yield item['text']
        cnt += 1
        if cnt > 5e3: break

def calc_compression_ratio():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    ds = load_dataset(DATASET_NAME, split='train')

    sum = 0
    max_len = 0
    mean_compression_ratio = 0
    for i, item in tqdm(enumerate(iter_func(ds))):
        ids = tokenizer.encode(item)
        sum += len(ids)
        mean_compression_ratio += len(item) / len(ids)
        max_len = max(max_len, len(ids))

        if i > 1e4: break

        if (i+1) % 1000 == 0:
            print(i+1, sum / (i+1), max_len, mean_compression_ratio / (i+1))

def test():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    ddsl = DistributedIteratorDataLoader(
        dataset=iter_func(load_dataset(DATASET_NAME, split='train')),
        batch_size=16,
        tokenizer=tokenizer,
        max_length=4096,
        mean_compression_ratio=4.23,
    )

    tmp = 0
    cnt = 0
    for i, data in enumerate(ddsl):
        tmp = tmp + ops.sum(data['attention_mask']).item()
        cnt += 1
        print('i=',i)

    print(tmp / cnt / 16 / 4096)

bms.init_distributed()
test()