
from os.path import join

import torch

from deept.utils.log import write_number_to_file


__LR_SCHEDULER_DICT__ = {}


def register_lr_scheduler(name):
    def register_lr_scheduler_fn(cls):
        if name in __LR_SCHEDULER_DICT__:
            raise ValueError(f'Lr scheduler {name} already registered!')
        __LR_SCHEDULER_DICT__[name] = cls
        return cls

    return register_lr_scheduler_fn

def create_lr_scheduler_from_config(lr_scheduler_config, optimizer, name):
    return __LR_SCHEDULER_DICT__[lr_scheduler_config['lr_type']].create_from_config(lr_scheduler_config, optimizer, name)

def get_all_lr_scheduler_keys():
    return list(__LR_SCHEDULER_DICT__.keys())


class LR_Scheduler:
    """
    A simplified form of torch.optim.lr_scheduler.
    """

    def __init__(self,
        optimizer, name,
        **kwargs
    ):
        self.name = name
        self.per_step = True
        self.optimizer = optimizer

        if 'per_step' in kwargs:
            self.per_step = kwargs['per_step']

        self.step_count = 0
        self.lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        if self.per_step:
            self.update()
        self.__write_lrs_to_file()

    def epoch(self):
        if not self.per_step:
            self.update()

    def update(self):
        self.step_count +=1
        lrs = self.get_lrs()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        self.lrs = lrs

    def __write_lrs_to_file(self):
        write_number_to_file('lrs', self.lrs)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


@register_lr_scheduler('Constant')
class Constant(LR_Scheduler):

    def __init__(self, optimizer, *args, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.optimizer = optimizer

        super().__init__(optimizer, *args)

    @staticmethod
    def create_from_config(config, optimizer, *args):
        return Constant(
            optimizer, *args,
            lr = config['lr'],
        )

    def get_lrs(self):
        return [self.lr for param_group in self.optimizer.param_groups] 


@register_lr_scheduler('CosineAnnealing')
class CosineAnnealingLR(LR_Scheduler):

    def __init__(self, optimizer, *args, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.tmax
        )

        super().__init__(optimizer, *args, **kwargs)

    @staticmethod
    def create_from_config(config, optimizer, *args):
        return CosineAnnealingLR(optimizer, *args,
            tmax = config['tmax'],
            per_step = config['per_step', False]
        )

    def get_lrs(self):
        self.scheduler.step()
        return self.scheduler.get_last_lr()


@register_lr_scheduler('OneCycle')
class OneCycleLR(LR_Scheduler):

    def __init__(self, optimizer, *args, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.total_steps
        )

        super().__init__(optimizer, *args, **kwargs)

    @staticmethod
    def create_from_config(config, optimizer, *args):
        return OneCycleLR(optimizer, *args,
            max_lr = config['max_lr'],
            total_steps = config['total_steps'],
            per_step = config['per_step', False]
        )

    def get_lrs(self):
        self.scheduler.step()
        return self.scheduler.get_last_lr()


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