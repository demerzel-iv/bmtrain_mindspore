# BMTrain on mindspore

## 简介

BMTrain是一个高效的大模型训练工具包，可用于训练拥有数百亿参数的大模型。它能够以分布式方式进行训练，同时保持代码像单机训练一样简洁。此代码库为BMTrain在Mindspore框架上的适配版本。

## 安装

- 通过代码安装：下载源代码并在目录下运行`pip install .`
- 按照以下说明对mindspore代码进行修改

### 代码修改

依赖mindspore==2.5.0

需要对mindspore代码进行修改：

**`mindspore.common.parameter.Parameter.__new__`**

这个函数中写死了返回的类型，导致`Parameter`类不能被正确地继承，需要修改如下：

将`Parameter._get_base_class(input_class)`改为

```python
    @classmethod
    def _get_base_class(cls, input_class):
        input_class_name = cls.__name__
        if input_class_name in cls._base_type:
            new_type = cls._base_type.get(input_class_name)
        else:
            new_type = type(input_class_name, (cls, input_class), {})
            cls._base_type[input_class_name] = new_type
        return new_type
```

然后将`Parameter.__new__`的第4，5行从

```python
        input_class, *class_init_args = Parameter._get_parameter_new_args(default_input, rc, init_param)
        new_type = Parameter._get_base_class(input_class)
```

改为

```python
        input_class, *class_init_args = cls._get_parameter_new_args(default_input, rc, init_param)
        new_type = cls._get_base_class(input_class)
```

## 使用说明

### 步骤 1: 启用 BMTrain

首先，你需要在代码开头初始化 BMTrain，在代码开头使用 **init_distributed**。

```python
import bmtrain_mindspore as bms
bms.init_distributed()
```

### 步骤 2: 使用 ZeRO 优化

使用ZeRO优化需要对模型代码进行简单替换：

- `mindspore.nn.Cell` -> `bmtrain_mindspore.DistributedModule`
- `mindspore.Parameter` -> `bmtrain_mindspore..DistributedParameter`

或者你可以使用`bmtrain_mindspore.model_center.layer`中已有的组建搭建自己的模型。

另外，在`bmtrain_mindspore.model_center.model`中，我们已经实现了Llama2和Deepseek-v2的`bmtrain_mindspore`版本，可以通过`convert_utils`目录下的脚本将huggingface版本的模型权重转换为`bmtrain_mindspore`版本。

### 步骤 3: 运行分布式训练代码

请参考以下脚本

```bash
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <n_works> <exe_file>"
    exit 1
fi

n_works=$1
exe_file=$2

msrun --bind_core True \
     --master_port 8129\
     --worker_num ${n_works} --local_worker_num ${n_works} ${exe_file} 
```

## 样例

在`examples/train-small-llama`中，我们提供了一个小规模的从零训练llama模型的样例。

### 步骤1：准备数据

请确保能够连接到`huggingface.co`以下载训练数据，或者使用镜像网站：`export HF_ENDPOINT=https://hf-mirror.com`

### 步骤2：准备tokenizer

直接下载huggingface版本的llama2 tokenizer即可`

### 步骤3：训练模型

在`pretrain.py`中修改tokenizer及输出路径，并在`pretrain.py`中，然后在`example`目录下运行`bash run.sh 8 train-small-llama/pretrain.py`

### 步骤4：使用模型生成

在`pretrain.py`中调用`generate()`，然后在`example`目录下运行`bash run.sh 1 train-small-llama/pretrain.py`

然后查看模型生成结果即可
