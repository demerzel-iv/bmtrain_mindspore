import os
import json

from typing import Type, TypeVar

ConfigType = TypeVar('ConfigType', bound='Config')

class Config:
    """
    The base class of all model config classes.
    """
    @classmethod
    def from_pretrained(cls: Type[ConfigType], pretrained_model_path: str, **kwargs) -> ConfigType:
        return cls.from_json_file(os.path.join(pretrained_model_path, 'config.json'), **kwargs)

    @classmethod
    def from_json_file(cls: Type[ConfigType], json_file: str, **kwargs) -> ConfigType:
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        for key in kwargs:
            config_dict[key] = kwargs[key]
        return cls(**config_dict)