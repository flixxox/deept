
from os.path import join

import torch

from deept.util.debug import write_number_to_file

__LR_SCHEDULER_DICT__ = {}

def register_lr_scheduler(name):
    def register_lr_scheduler_fn(cls):
        if name in __LR_SCHEDULER_DICT__:
            raise ValueError(f'Lr scheduler {name} already registered!')
        __LR_SCHEDULER_DICT__[name] = cls
        return cls

    return register_lr_scheduler_fn

def create_lr_scheduler_from_config(config, optimizer):
    if config['lr_scheduler'] in __LR_SCHEDULER_DICT__:
        return __LR_SCHEDULER_DICT__[config['lr_scheduler']].create_from_config(config, optimizer)
    else:
        raise ValueError(f'Error! Unrecognized lr scheduler {config["lr_scheduler"]}!')


class LR_Scheduler:
    """
    A simplified form of torch.optim.lr_scheduler.
    """

    def __init__(self, optimizer):

        self.optimizer = optimizer
        self.step_count = 0
        self.lrs = [0. for _ in self.optimizer.param_groups]
        self.step() # Do an initial step

    def step(self):
        self.step_count +=1
        lrs = self.get_lrs()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        self.lrs = lrs
        self.__write_lrs_to_file()

    def __write_lrs_to_file(self):
        write_number_to_file('lrs', self.lrs)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


@register_lr_scheduler('Warmup')
class Warmup(LR_Scheduler):

    def __init__(self, optimizer, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.optimizer = optimizer

        super().__init__(optimizer)

    @staticmethod
    def create_from_config(config, optimizer):
        return Warmup(
            optimizer,
            model_dim = config['model_dim'],
            warmup = config['warmup_steps', 4000],
            lr_scale = config['warmup_lr_scale', 2.0]
        )

    def get_lrs(self):

        w  = self.warmup
        D  = self.model_dim
        s  = self.step_count
        e1 = -0.5
        e2 = -1.5

        return [(D ** e1) * min(s ** e1, s * w ** e2) * self.lr_scale]