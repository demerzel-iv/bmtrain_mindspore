from .base_model import BaseModel
from .config import LlamaConfig

class Llama(BaseModel):
    _CONFIG_TYPE = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__()