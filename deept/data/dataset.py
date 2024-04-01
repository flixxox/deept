

from functools import partial

import torch

from deept.utils.debug import my_print
from deept.utils.globals import Settings, Context


__DATASETS__ = {}

def register_dataset(name):
    def register_dataset_fn(cls):
        if name in __DATASETS__:
            raise ValueError(f'Dataset {name} already registered!')
        __DATASETS__[name] = cls
        return cls
    return register_dataset_fn

def create_dataset_from_config(config, is_train, key):
    dataset = __DATASETS__[config[key]].create_from_config(config, is_train)
    return dataset

def get_all_dataset_keys():
    return list(__DATASETS__.keys())