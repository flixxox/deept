
from deept.util.globals import Context

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


@register_postprocessing_fn('mt_postprocess')
def mt_postprocesser(name, value):

    from deept.util.globals import Context

    if 'src' in name:
        vocab = Context['vocab_src']
    elif (
        'tgt' in name or 
        'result' in name or
        'beam' in name or
        'out' in name):
        vocab = Context['vocab_tgt']
    else:
        raise ValueError(f'Unrecognized tensor name {name} for postprocessing!')

    value = value.numpy().tolist()

    processed = []

    for entry in value:
        entry = vocab.detokenize(entry)
        entry = vocab.remove_padding(entry)
        entry = ' '.join(entry)
        entry = entry.replace(vocab.EOS_TEXT, '')
        entry = entry.strip()
        processed.append(entry)

    return processed