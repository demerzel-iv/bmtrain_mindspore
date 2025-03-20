import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from time import time
from datasets import load_dataset
from mindnlp.core import ops
from mindspore import mint, Tensor, Parameter
from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer
from bmtrain_mindspore.model_center.model.llama import Llama, LlamaConfig
from bmtrain_mindspore.utils import Timer

from data_utils import DistributedDataLoader

from mindspore import nn
from mindspore.experimental import optim

class WarmupStableDecayLRScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        warmup_steps=100,
        decay_start=4000,
        total_iters=5000,
        lr=1e-5
    ):
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.total_iters = total_iters
        self.lr = lr
        super(WarmupStableDecayLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup
            lr_ret = self.lr * (self.last_epoch + 1) / self.warmup_steps
        elif self.last_epoch < self.decay_start:
            # Stable
            lr_ret = self.lr
        else:
            # Decay
            decay_iters = self.total_iters - self.decay_start
            lr_ret = self.lr * max(0, 1 - (self.last_epoch - self.decay_start) / decay_iters)

        return [lr_ret] * len(self._last_lr)


def random_init_model(model: Llama):
    for name, param in model.parameters_and_names():
        assert isinstance(param, Parameter)
        if 'layernorm.weight' in name:
            param.set_data(Tensor(
                np.ones(*param.shape),
                dtype=param.dtype
            ))
        else:
            param.set_data(Tensor(
                np.random.randn(*param.shape) * 0.03,
                dtype=param.dtype
            ))

def generate():
    tokenizer_path = '/root/thunlp/data/Llama-2-7b-tokenizer'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig.from_json_file('./small_llama.json', dtype='bf16')
    model = Llama(config)
    bms.load(model, '/root/thunlp/data/test_model/model.safetensors')

    input_ids = Tensor([[tokenizer.bos_token_id]])
    key_values = None
    tok_list = (input_ids,)
    stream = ''

    temperature = 0.7

    for i in range(1000):
        _, key_values, logits = model.construct(
            input_ids=input_ids,
            output_logits=True,
            use_cache=True,
            past_key_values=key_values,
        )
        prob = ops.softmax(logits.reshape(-1).astype(ms.float32) / temperature).numpy()
        next_tok_id = np.random.choice(logits.shape[-1], p=prob)

        input_ids = Tensor([[next_tok_id]])
        tok_list += (input_ids,)

        res = ops.cat(tok_list, dim=-1)
        new_stream = tokenizer.decode(res[0])
        print(new_stream[len(stream):], end='', flush=True)
        stream = new_stream

        if input_ids.item() == tokenizer.eos_token_id:
            break

    print('')
    print(tokenizer.convert_ids_to_tokens(res[0]))


def valid():
    tokenizer_path = '/root/thunlp/data/Llama-2-7b-tokenizer'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = list(map(
        lambda item: item['text'].replace('<|endoftext|>', '</s>'),
        load_dataset("fhswf/TinyStoriesV2_cleaned", split='test')
    ))
    valid_loader = DistributedDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=60,
        max_length=512 + 1,
    )

    config = LlamaConfig.from_json_file('./small_llama.json', dtype='bf16')
    model = Llama(config)
    bms.load(model, '/root/thunlp/data/test_model/model.safetensors')

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

    for iter, input_data in enumerate(valid_loader):
        # prepare input
        input_ids = input_data['input_ids'][:, :-1]
        attention_mask = input_data['attention_mask'][:, :-1]
        labels = input_data['input_ids'][:, 1:].copy().astype(ms.int32)
        labels[input_data['attention_mask'][:, 1:] == 0] = -100

        # optim step
        loss = forward_fn(input_ids, attention_mask, labels)

        print('valid iter {}/{} | valid | loss: {:.3f} | mem: {:.1f}GB'.format(
            iter, len(valid_loader), float(loss),
            ms.runtime.max_memory_allocated() / (2**30),
        ), flush=True)


def train():
    tokenizer_path = '/root/thunlp/data/Llama-2-7b-tokenizer'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = list(map(
        lambda item: item['text'].replace('<|endoftext|>', '</s>'),
        load_dataset("fhswf/TinyStoriesV2_cleaned", split='train')
    ))
    train_loader = DistributedDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=80,
        max_length=512 + 1,
    )
    print('length of dataset: {}'.format(len(train_loader)))

    config = LlamaConfig.from_json_file('./small_llama.json', dtype='bf16')
    model = Llama(config)
    random_init_model(model)

    lr = 1e-3
    optimizer = mint.optim.Adam(model.trainable_params(), eps=1e-5)
    scheduler = WarmupStableDecayLRScheduler(
        optimizer=optimizer,
        warmup_steps=100,
        decay_start=len(train_loader) * 0.8,
        total_iters=len(train_loader),
        lr=lr,
    )
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

        print('iter {} | lr: {:.1e} | loss: {:.3f} | mem: {:.1f}GB/{:.1f}GB | time: {:.2f}s+{:.2f}s'.format(
            iter, float(scheduler.get_lr()[0]), float(loss),
            ms.runtime.max_memory_allocated() / (2**30),
            ms.runtime.memory_allocated() / (2**30),
            train_timer.elapsed_time,
            time() - lst_time - train_timer.elapsed_time,
        ), flush=True)
        lst_time = time()

    bms.save(model, '/root/thunlp/data/test_model/model.safetensors')


bms.init_distributed()

train()
valid()
generate()