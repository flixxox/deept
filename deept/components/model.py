import torch.nn as nn

__MODEL_DICT__ = {}

def register_model(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn

def create_model_from_config(config):
    from deept.util.debug import my_print
    model = __MODEL_DICT__[config['model']].create_from_config(config)
    check_model(model)
    my_print(f'Initializing model weights!')
    if hasattr(model, 'init_weights'):
        model.init_weights()
    return model

def check_model(model):

    assert hasattr(model, 'input_keys'), """To train your model with DeepT, give it the attribute 'input_keys'.
        'input_keys' can be read from the config during model.create_from_config(config). It is expected to be a list
        of data keys which are fed to the __call__ function of your model."""

    assert isinstance(model.input_keys, list) 

def get_all_model_keys():
    return list(__MODEL_DICT__.keys())