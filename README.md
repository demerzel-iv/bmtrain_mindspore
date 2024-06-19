## 环境说明

版本： mindspore==2.3.0rc1

代码修改：

### `mindspore.common.parameter.Parameter.__new__`

这个函数中写死了返回的类型，导致`Parameter`类不能被正确地继承，因此修改如下：

将`Parameter._get_base_class(input_class)`改为：

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

然后将`Parameter.__new__`的第3，4行从

```python
        input_class, *class_init_args = Parameter._get_parameter_new_args(default_input, rc)
        new_type = Parameter._get_base_class(input_class)
```

改成

```python
        input_class, *class_init_args = cls._get_parameter_new_args(default_input, rc)
        new_type = cls._get_base_class(input_class)
```

## 已完成

- 分布式模型存储
- 分布式模型的保存/加载
- 前向/反向传播计算正确性验证
- 模型组件：Linear, Embedding, LayerNorm, Attention, FeedForward
- 完整模型：组合模型插件实现
- huggingface的ckpt到mindspore的ckpt的转换脚本

## TODOLIST

- 迁移LLaMA模型(LLaMA2-7B)，验证正确性
    - 正确性验证：前向/反向传播
- optimizer
    - Adam优化器
    - 管理器
- ZeRO优化
- 验证模型训练能正确收敛