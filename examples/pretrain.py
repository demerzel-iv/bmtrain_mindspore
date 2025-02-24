import numpy as np
import mindspore as ms
import bmtrain_mindspore as bms

from mindspore import mint, ops
from datasets import load_dataset
from mindspore import Tensor, Parameter
from mindspore.mint.nn import CrossEntropyLoss
from mindnlp.transformers import PreTrainedTokenizerFast, AutoTokenizer
from bmtrain_mindspore.model_center.model.llama import Llama, LlamaConfig

from utils import DistributedDataLoader


from mindspore import nn
from mindspore.experimental import optim
class WarmupDecayLRScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, warmup_steps=1000, total_iters=100000, lr=1e-5):
        self.total_iters = total_iters
        self.warmup_steps = warmup_steps
        self.lr = lr
        super(WarmupDecayLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_ret = self.lr * (self.last_epoch + 1) / self.warmup_steps
        else:
            lr_ret = self.lr * (1 - (self.last_epoch - self.warmup_steps) / (self.total_iters - self.warmup_steps))

        return [lr_ret] * len(self._last_lr)

        #if self.last_epoch == 0:
        #    return [lr * self.factor for lr in self._last_lr]
        #if self.last_epoch != self.total_iters:
        #    return [lr * 1. for lr in self._last_lr]
        #return [lr / self.factor for lr in self._last_lr]


def count():
    tokenizer_path = '/home/hanxu/lyq/data/Llama-2-7b-ms'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = list(map(
        lambda item: item['text'].replace('<|endoftext|> ', '</s>'),
        load_dataset("fhswf/TinyStoriesV2_cleaned", split='test'),
    ))

    from tqdm import tqdm
    cnt = []
    for line in tqdm(dataset):
        cnt.append(len(tokenizer.encode(line)))
    cnt = np.array(cnt)
    for i in range(4):
        print(i, np.sum(cnt // 512 == i))
    print('long', np.sum(cnt // 512 >= 4))

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

def test_init():
    bms.init_distributed(device_list=[4, 5, 6, 7])

    tokenizer_path = '/home/hanxu/lyq/data/Llama-2-7b-ms'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path)

    model = Llama.from_pretrained(tokenizer_path)
    #config = LlamaConfig.from_json_file('./small_llama.json')
    #model = Llama(config)
    #random_init_model(model)

    #for name, param in model.parameters_and_names():
    #    print(name, np.array(param).std())

    loss_func = CrossEntropyLoss()

    s = "The sun sets in the west, painting the sky with"
    x = tokenizer.encode(s, return_tensors='ms') # bs, n

    y = x[:, 1:].copy().astype(ms.int32)

    _, _, logits = model.construct(input_ids=x[:, :-1], output_logits=True)

    labels = y.reshape(-1)
    logits = logits.reshape(-1, logits.shape[-1])

    print(logits)
    print(y)
    print(loss_func(logits, labels))

def train():
    bms.init_distributed(device_list=[4, 5, 6, 7])

    tokenizer_path = '/home/hanxu/lyq/data/Llama-2-7b-ms'
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = list(map(
        lambda item: item['text'].replace('<|endoftext|>', '</s>'),
        load_dataset("fhswf/TinyStoriesV2_cleaned", split='test')
    ))

    loss_func = CrossEntropyLoss()

    train_loader = DistributedDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=513,
    )

    #config = LlamaConfig.from_json_file('./small_llama.json')
    #model = Llama(config)
    #random_init_model(model)
    model = Llama.from_pretrained(tokenizer_path)

    lr = 1e-5
    optimizer = mint.optim.Adam(model.trainable_params(), lr=0.0)
    scheduler = WarmupDecayLRScheduler(
        optimizer=optimizer,
        warmup_steps=300,
        lr = lr,
    )

    def forward_fn(input_ids, attention_mask, labels):
        _, _, logits = model.construct(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_logits=True,
        )
        labels = labels.reshape(-1)
        logits = logits.reshape(-1, logits.shape[-1])
        print(logits)
        print(labels)
        loss = loss_func.construct(logits, labels)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    for iter, input_data in enumerate(train_loader):
        input_ids = input_data['input_ids'][:, :-1]
        attention_mask = input_data['attention_mask'][:, :-1]

        labels = input_data['input_ids'][:, 1:].copy().astype(ms.int32)
        labels[input_data['attention_mask'][:, 1:] == 0] = -100

        loss, grads = grad_fn(input_ids, attention_mask, labels)
        grads = tuple(gd / bms.world_size() for gd in grads)
        print(grads)
        optimizer.construct(grads)
        scheduler.step()

        print(iter, loss)
        print('memory occupation - {}'.format(ms.runtime.memory_allocated()))
        for name, param in model.parameters_and_names():
            print(name, np.array(param).std())
        print('=======\n\n\n')

        #_, _, logits = model.construct(
        #    input_ids=input_ids,
        #    attention_mask=attention_mask,
        #    output_logits=True,
        #)

        #labels = labels.reshape(-1)
        #logits = logits.reshape(-1, logits.shape[-1])
        #loss = loss_func.construct(logits, labels)

        #print(loss)

        if iter == 2:
            break

train()
#test_init()