
import torch

__OPTIMIZER_DICT__ = dict()

def create_optimizer_from_config(config, model_params):
    if config['optimizer'] in __OPTIMIZER_DICT__:
        return __OPTIMIZER_DICT__[config['optimizer']].create_from_config(config, model_params)
    else:
        raise ValueError(f'Error! Unrecognized optimizer {config["optimizer"]}!')

def register_optimizer(name):
    def register_optimizer_fn(cls):
        if name in __OPTIMIZER_DICT__:
            raise ValueError(f'Optimizer {name} already registered!')
        __OPTIMIZER_DICT__[name] = cls
        return cls

    return register_optimizer_fn

@register_optimizer('Adam')
class Adam:
    @staticmethod
    def create_from_config(config, model_params):
        return torch.optim.Adam(
            model_params,
            betas=(config['adam_beta1', 0.9], config['adam_beta2', 0.98]),
            eps=config['adam_eps', 1e-9]
        )