

from deept.utils.config import Config
from deept.utils.globals import Settings

__DATALOADER__ = {}

def register_dataloader(name):
    def register_dataloader_fn(cls):
        if name in __DATALOADER__:
            raise ValueError(f'Dataloader {name} already registered!')
        __DATALOADER__[name] = cls
        return cls
    return register_dataloader_fn

def create_dataloader_from_config(config, dataset, is_train=False, num_worker_overwrite=None):
    if num_worker_overwrite is not None:
        num_workers = num_worker_overwrite
    else:
        num_workers = config['dataloader_workers']
    dataloader = __DATALOADER__[config['dataloader']].create_from_config(config,
        dataset,
        is_train,
        num_workers
    )
    return dataloader

def get_all_dataloader_keys():
    return list(__DATALOADER__.keys())
