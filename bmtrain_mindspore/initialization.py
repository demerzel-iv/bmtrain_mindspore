import os
import mindspore as ms

from .global_var import config
from typing import List

def init_distributed(
        synchronous_execution: bool = False,
        device_list: List[int] = None,
):
    if config['initialized']:
        return 

    ms.set_context(mode=ms.PYNATIVE_MODE)

    rank = int(os.environ['RANK_ID'])
    if device_list != None:
        ms.set_device('Ascend', device_list[rank])
    else:
        ms.set_device('Ascend', rank)
    if synchronous_execution:
        ms.runtime.launch_blocking()

    ms.communication.init('hccl')
    world_size = ms.communication.get_group_size()

    config['initialized'] = True
    config['rank'] = rank
    config['world_size'] = world_size