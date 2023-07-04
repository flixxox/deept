import torch.nn as nn

__MODEL_DICT__ = dict()

def register_model(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn

def create_model_from_config(config, vocab_src, vocab_tgt):

    if config['model'] in __MODEL_DICT__:
        from deept.util.debug import my_print
        model = __MODEL_DICT__[config['model']].create_from_config(config, vocab_src, vocab_tgt)
        my_print(f'Initializing model weights!')
        model.init_weights()
        return model
    else:
        raise ValueError(f'Error! Unrecognized model {config["model"]}!')


class MTModel(nn.Module):

    def __init__(self):
        super().__init__()

    def init_weights_from_checkpoint(self, model_params):
        pass
