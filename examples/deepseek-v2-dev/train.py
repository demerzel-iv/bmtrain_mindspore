import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import shutil
import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from time import time
from typing import Iterable
from datasets import load_dataset
from mindspore import mint, Tensor, Parameter, ops, nn
from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer
from bmtrain_mindspore.model_center.model.deepseek_v2 import DeepseekV2
from bmtrain_mindspore.utils import Timer
from bmtrain_mindspore.model_center.lr_scheduler import WarmupStableDecayLRScheduler

from data_utils import DistributedIteratorDataLoader

LLM_PATH = '/root/thunlp/data/DeepSeek-V2-Lite-ms'
CHECKPOINT_ROOT = '/root/thunlp/data/moe_checkpoints'

def save_checkpoint(
    iter: int,
    model: DeepseekV2,
    optimizer: nn.Cell,
    data_loader: DistributedIteratorDataLoader,
    num_token_passed: int,
):
    save_path = os.path.join(CHECKPOINT_ROOT, f"iter_{iter}")
    rank = bms.rank()
    os.makedirs(save_path, exist_ok=True)
    shutil.copy(
        os.path.join(LLM_PATH, "config.json"),
        os.path.join(save_path, "config.json"),
    )
    bms.save(
        model,
        os.path.join(save_path, "mindspore_model.safetensors"),
    )
    ms.save_checkpoint(
        optimizer,
        os.path.join(save_path, f'optimizer_rank{rank}.ckpt'),
    )
    metadata = {
        "iter": iter,
        "data_loader_start_index": data_loader.start_index,
        "num_token_passed": num_token_passed,
    }
    with open(os.path.join(save_path, f'metadata_rank{rank}.json'), "w") as f:
        json.dump(metadata, f)
    print(f"Checkpoint saved at {save_path}")

def init_from_checkpoint(
    checkpoint_iter: int = None,
):
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(LLM_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    rank = bms.rank()

    if checkpoint_iter is None:
        checkpoint_iter = -1
        data_loader_start_index = 0
        checkpoint_path = LLM_PATH
        num_token_passed = 0
    else:
        checkpoint_path = os.path.join(CHECKPOINT_ROOT, f"iter_{checkpoint_iter}")
        with open(os.path.join(checkpoint_path, f'metadata_rank{rank}.json'), "r") as f:
            metadata = json.load(f)
        data_loader_start_index = metadata["data_loader_start_index"]
        num_token_passed = metadata["num_token_passed"]

    # Load the dataset
    train_loader = DistributedIteratorDataLoader(
        dataset=load_dataset('venketh/SlimPajama-62B', split='train'),
        tokenizer=tokenizer,
        batch_size=6,
        max_length=4096 + 1, # +1 for next token prediction
        mean_compression_ratio=4.23, # the mean compression ratio of this dataset computed by me
        start_index=data_loader_start_index,
    )

    model = DeepseekV2.from_pretrained(checkpoint_path)
    model.set_train(True)

    lr = 1e-5
    optimizer = mint.optim.Adam(
        model.trainable_params(),
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    # this code prevents scheduler from raising error when loading checkpoint
    for pg in optimizer.param_groups:
        pg.setdefault('initial_lr', 1e9)
    # end
    opt_path = os.path.join(checkpoint_path, f'optimizer_rank{rank}.ckpt')
    if os.path.exists(opt_path):
        print(f"Loading optimizer from {opt_path}")
        ms.load_checkpoint(opt_path, optimizer)
    else:
        print(f"Optimizer checkpoint not found at {opt_path}, using default optimizer")

    scheduler = WarmupStableDecayLRScheduler(
        optimizer=optimizer,
        last_epoch=checkpoint_iter,
        warmup_steps=100,
        decay_start=float('inf'),
        total_iters=float('inf'),
        lr=lr,
    )

    print(f"Checkpoint loaded from {checkpoint_path}")
    start_iter = checkpoint_iter + 1

    return start_iter, model, optimizer, train_loader, scheduler, num_token_passed

def train():
    start_iter, model, optimizer, train_loader, \
        scheduler, num_token_passed = init_from_checkpoint(checkpoint_iter=None)

    loss_func = mint.nn.CrossEntropyLoss()

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
    sum_reduce = ops.AllReduce(op=ops.ReduceOp.SUM)

    for iter, input_data in enumerate(train_loader, start=start_iter):
        # prepare input
        input_ids = input_data['input_ids'][:, :-1]
        attention_mask = input_data['attention_mask'][:, :-1]
        labels = input_data['input_ids'][:, 1:].copy().to(ms.int32)
        labels[input_data['attention_mask'][:, 1:] == 0] = -100

        num_token_passed += sum_reduce(ops.sum(labels != -100).to(ms.int32)).item()

        # optim step
        with Timer(print_to_screen=False) as train_timer:
            loss, grads = grad_fn(input_ids, attention_mask, labels)
            grads = tuple(gd / bms.world_size() for gd in grads)
            optimizer.construct(grads)
            scheduler.step()

        loss_to_print = sum_reduce(loss) / bms.world_size()
        print(
            f"iter {iter} | lr: {float(scheduler.get_lr()[0]):.1e} | "
            f"loss: {float(loss_to_print):.3f} | aux_loss: {float(model.aux_loss):.3f} | "
            f"#token passed: {num_token_passed:.2e} | "
            f"mem: {ms.runtime.max_memory_allocated() / (2**30):.1f}GB/"
            f"{ms.runtime.memory_allocated() / (2**30):.1f}GB | "
            f"time: {train_timer.elapsed_time:.2f}s+"
            f"{time() - lst_time - train_timer.elapsed_time:.2f}s",
            flush=True
        )
        lst_time = time()

        if (iter + 1) % 500 == 0:
            with Timer('save_checkpoint'):
                save_checkpoint(
                    iter=iter,
                    model=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    num_token_passed=num_token_passed,
                )

bms.init_distributed()

train()