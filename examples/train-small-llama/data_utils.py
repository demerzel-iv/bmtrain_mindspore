import numpy as np
import bmtrain_mindspore as bms

from datasets import load_dataset
from typing import Iterable, List
from mindnlp.transformers import PreTrainedTokenizerBase

class DistributedDataLoader:
    def __init__(self,
            dataset: List[str],
            tokenizer: PreTrainedTokenizerBase,
            batch_size: int,
            max_length: int,
            shuffle: bool = True,
            seed=0,
        ):
        np.random.seed(seed)
        self.dataset = []
        dataset_size = len(dataset) // bms.world_size()
        idx = np.random.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
        for i in range(dataset_size):
            self.dataset.append(dataset[idx[i*bms.world_size() + bms.rank()]])

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, i):
        start = i * self.batch_size
        end = (i + 1) * self.batch_size
        data = self.tokenizer.batch_encode_plus(
            self.dataset[start:end], 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_attention_mask=True,
            return_tensors='ms',
        )
        return data

    def __len__(self):
        return len(self.dataset) // self.batch_size