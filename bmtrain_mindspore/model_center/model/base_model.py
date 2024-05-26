from typing import Type

from mindspore.nn import Cell

from .config import Config

class BaseModel(Cell):
    _CONFIG_TYPE: Type[Config]

    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)