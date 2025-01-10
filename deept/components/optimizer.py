
import torch

from deept.utils.config import Config
from deept.components.lr_scheduler import create_lr_scheduler_from_config

__OPTIMIZER_DICT__ = {}

def create_optimizers_and_lr_schedulers_from_config(config, model):
    def __create_optimizer_and_lr_scheduler(optimizer_config, param_groups):
        param_groups_config = optimizer_config['param_groups']
        lr_scheduler_config = optimizer_config['lr_scheduler']

        if (not isinstance(param_groups_config, list)
                or len(param_groups_config) < 1):
            raise ValueError((
                f'The config "param_groups" of an optimizer must be a list with at least one element! '
                f'Supplied {param_groups_config}!'
            ))

        lr_scheduler_name = ['lr_scheduler']

        for i, param_group in enumerate(param_groups_config):
            param_group_name = param_group['name']
            lr_scheduler_name.append(param_group_name)
            param_groups_config[i]['params'] = param_groups[param_group_name]
        
        lr_scheduler_name = '_'.join(lr_scheduler_name)

        optimizer = create_optimizer_from_config(optimizer_config, param_groups_config)
        lr_scheduler = create_lr_scheduler_from_config(lr_scheduler_config, optimizer, lr_scheduler_name)

        return optimizer, lr_scheduler

    optimizers = []
    lr_schedulers = []
    optimizer_configs = config['optimizers']

    if isinstance(optimizer_configs, Config):
        optimizer_configs = [optimizer_configs]

    if not isinstance(optimizer_configs, list):
        raise ValueError(f'Optimizers must be either a list or dict! Supplied: {optimizer_configs}!')

    if hasattr(model, 'param_groups'):
        param_groups = model.param_groups
    else:
        param_groups = {'all': model.parameters()}
    
    for optimizer_config in optimizer_configs:
        optimizer, lr_scheduler = __create_optimizer_and_lr_scheduler(optimizer_config, param_groups)
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

    return optimizers, lr_schedulers

def create_optimizer_from_config(optimizer_config, param_groups_config):
    return __OPTIMIZER_DICT__[optimizer_config['optim_type']].create_from_config(optimizer_config, param_groups_config)

def register_optimizer(name):
    def register_optimizer_fn(cls):
        if name in __OPTIMIZER_DICT__:
            raise ValueError(f'Optimizer {name} already registered!')
        __OPTIMIZER_DICT__[name] = cls
        return cls
    return register_optimizer_fn

def get_all_optimizer_keys():
    return list(__OPTIMIZER_DICT__.keys())


@register_optimizer('Adam')
class Adam:
    @staticmethod
    def create_from_config(optimizer_config, param_groups_config):
        return torch.optim.Adam(
            [{
                'params': group['params'],
                'weight_decay': group['weight_decay'],
            } for group in param_groups_config]
        )