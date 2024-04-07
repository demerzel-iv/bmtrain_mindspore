from .global_var import config

def print_rank(*args, rank=0, **kwargs):
    if config['rank'] == rank:
        print(*args, **kwargs)