import mindspore as ms

from typing_extensions import TypedDict

class ConfigMap(TypedDict):
    rank : int
    world_size : int
    initialized : bool

config = ConfigMap(rank=0, world_size=1, initialized=False)

def rank():
    return config['rank']

def world_size():
    return config['world_size']
