import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from datasets import load_dataset
from typing import Iterable, List
from itertools import tee
from mindnlp.transformers import PreTrainedTokenizerBase
from mindspore import Tensor, ops

SENTENCE_BOUNDARY_MARKER = '<｜end▁of▁sentence｜><｜begin▁of▁sentence｜>'

def mark_last(iterable):
    """
    Marks the last element of an iterable with a boolean flag and includes the index.
    """
    try:
        prev = next(iterable)
    except StopIteration:
        return
    for cur in iterable:
        yield prev, False
        prev = cur
    yield prev, True  # the last element is marked as True

class DistributedIteratorDataLoader:
    def __init__(self,
            dataset: Iterable[str],
            tokenizer: PreTrainedTokenizerBase,
            batch_size: int,
            max_length: int,
            mean_compression_ratio: float = 1.0,
        ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mean_compression_ratio = mean_compression_ratio

        # Calculate chunk size of text based on max_length and mean_compression_ratio
        self.chunk_size = int(self.max_length * self.mean_compression_ratio)
        self.rank = bms.rank()

    def chunk_text_by_length(self, text: str):
        """
        Splits the text into smaller chunks based on the max_length and mean_compression_ratio.
        """
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if i > 0 : # remove first word split by space
                chunk = chunk[chunk.find(' ')+1:]
            if i + self.chunk_size < len(text):
                chunk = chunk[:chunk.rfind(' ')]
            yield chunk
    
    def _generate_concated_text(self):
        """
        Concatenate the text from the dataset into chunks of max_length.
        """
        concat_text = ''
        for text in self.dataset:
            for chunk in self.chunk_text_by_length(text):
                if len(concat_text) > 0:
                    concat_text += SENTENCE_BOUNDARY_MARKER
                concat_text += chunk

                # multiply by 0.9 to avoid exceeding max_length too much
                if len(concat_text) > self.chunk_size * 0.9:
                    yield concat_text
                    concat_text = ''

        if len(concat_text) > 0:
            yield concat_text

    def _filter_by_rank(self, iterable: Iterable[str]):
        """
        Filters the iterable by rank.
        """
        for i, item in enumerate(iterable):
            if i % bms.world_size() == self.rank:
                yield item

    def __iter__(self):
        all_reduce = ops.AllReduce()
        batch = []
        for text, is_last in mark_last(
            self._filter_by_rank(
                self._generate_concated_text()
            )
        ):
            batch.append(text)
            if len(batch) == self.batch_size:
                yield self.tokenizer.batch_encode_plus(
                    batch,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='ms',
                )
                batch.clear()

            # Check if this is the last batch among all ranks
            signal = Tensor(int(is_last), dtype=ms.int32)
            sum_signal = all_reduce(signal)
            if sum_signal > 0:
                break