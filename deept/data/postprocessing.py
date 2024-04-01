
from deept.utils.globals import Context


__POSTPROCESSING_FUNCTIONS__ = {}


def register_postprocessing_fn(name):
    def decorator(fn):
        if name in __POSTPROCESSING_FUNCTIONS__:
            raise ValueError(f'Postprocessing function {name} already registered!')
        __POSTPROCESSING_FUNCTIONS__[name] = fn
        return fn
    return decorator

def get_postprocessing_fn(config):
    if config['postprocessing_fn'] in __POSTPROCESSING_FUNCTIONS__:
        return __POSTPROCESSING_FUNCTIONS__[config['postprocessing_fn']]
    else:
        raise ValueError(f'Error! Unrecognized postprocessing function {config["postprocessing_fn"]}!')

def get_all_postprocessing_keys():
    return list(__POSTPROCESSING_FUNCTIONS__.keys())