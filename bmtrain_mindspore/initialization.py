import os
import mindspore as ms

from .global_var import config

def init_distributed():
    ms.communication.init('hccl')
    world_size = ms.communication.get_group_size()
    rank = ms.communication.get_rank()

    ms.set_context(device_id=rank)

    config['initialized'] = True
    config['rank'] = rank
    config['world_size'] = world_size