import os
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from time import time
from typing import Iterable
from datasets import load_dataset
from mindspore import mint, Tensor, Parameter, ops
from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer
from bmtrain_mindspore.model_center.model.deepseek_v2 import DeepseekV2
from bmtrain_mindspore.utils import Timer
from bmtrain_mindspore.model_center.lr_scheduler import WarmupStableDecayLRScheduler

from data_utils import DistributedIteratorDataLoader

LLM_PATH = '/root/thunlp/data/DeepSeek-V2-Lite-ms'

def iter_func(text_items) -> Iterable[str]:
    cnt = 0
    for item in text_items:
        yield item['text']
        cnt += 1
        if cnt > 5e3: break

def train():
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(LLM_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = DistributedIteratorDataLoader(
        dataset=iter_func(load_dataset('venketh/SlimPajama-62B', split='train')),
        batch_size=1,
        tokenizer=tokenizer,

        #max_length=4096 + 1, # +1 for next token prediction
        max_length=1024 + 1, # +1 for next token prediction

        mean_compression_ratio=4.23, # the mean compression ratio of this dataset computed by me
    )

    model = DeepseekV2.from_pretrained(LLM_PATH)
    model.set_train(True)

    print('mem_loc before', ms.runtime.memory_allocated() / (2**30))

    lr = 1e-5
    optimizer = mint.optim.Adam(model.trainable_params())
    scheduler = WarmupStableDecayLRScheduler(
        optimizer=optimizer,
        warmup_steps=100,
        decay_start=float('inf'),
        total_iters=float('inf'),
        lr=lr,
    )
    loss_func = mint.nn.CrossEntropyLoss()

    print('mem_loc', ms.runtime.memory_allocated() / (2**30))

    def forward_fn(input_ids, attention_mask, labels):
        _, _, logits = model.construct(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_logits=True,
        )
        labels = labels.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = loss_func.construct(logits, labels)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    lst_time = time()

    for iter, input_data in enumerate(train_loader):
        # prepare input
        input_ids = input_data['input_ids'][:, :-1]
        attention_mask = input_data['attention_mask'][:, :-1]
        labels = input_data['input_ids'][:, 1:].copy().astype(ms.int32)
        labels[input_data['attention_mask'][:, 1:] == 0] = -100

        # optim step
        with Timer(print_to_screen=False) as train_timer:
            loss, grads = grad_fn(input_ids, attention_mask, labels)
            grads = tuple(gd / bms.world_size() for gd in grads)
            optimizer.construct(grads)
            scheduler.step()

        print('iter {} | lr: {:.1e} | loss: {:.3f} | aux_loss: {:.3f} | mem: {:.1f}GB/{:.1f}GB | time: {:.2f}s+{:.2f}s'.format(
            iter, float(scheduler.get_lr()[0]), float(loss), float(model.aux_loss),
            ms.runtime.max_memory_allocated() / (2**30),
            ms.runtime.memory_allocated() / (2**30),
            train_timer.elapsed_time,
            time() - lst_time - train_timer.elapsed_time,
        ), flush=True)
        lst_time = time()

    #bms.save(model, MODELFILE_SAVE_PATH)

bms.init_distributed()

train()