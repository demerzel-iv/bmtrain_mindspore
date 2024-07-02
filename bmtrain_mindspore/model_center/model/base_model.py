import os

from typing import Type, TypeVar
from mindspore.nn import Cell

from .config import Config
from ...store import load

ModelType = TypeVar('ModelType', bound='BaseModel')

class BaseModel(Cell):
    _CONFIG_TYPE: Type[Config]

    @classmethod
    def from_pretrained(cls: Type[ModelType], pretrained_model_path: str, config=None, **kwargs) -> ModelType:
        from ...global_var import rank
        print('here - {}'.format(rank()))
        if config == None:
            config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_path, **kwargs)
        print('there - {}'.format(rank()))
        model = cls(config)
        print('where - {}'.format(rank()))
        load(model, os.path.join(pretrained_model_path, 'mindspore_model.ckpt'), strict=True)
        return model