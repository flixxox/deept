

import torch
import torchdata.datapipes as dp

from deept.util.debug import my_print
from deept.util.globals import Settings, Context


__DP_DECODING__ = {}
__DP_PREPROCESSING__ = {}
__DP_COLLATE__ = {}
__DP_OVERWRITE__ = {}
__LEN_FN__ = {}


def register_dp_decoding(name):
    def register_dp_decoding_fn(cls):
        if name in __DP_DECODING__:
            raise ValueError(f'Decoding datapipe {name} already registered!')
        __DP_DECODING__[name] = cls
        return cls
    return register_dp_decoding_fn

def register_dp_preprocessing(name):
    def register_dp_preprocessing_fn(cls):
        if name in __DP_PREPROCESSING__:
            raise ValueError(f'Preprocessing datapipe {name} already registered!')
        __DP_PREPROCESSING__[name] = cls
        return cls
    return register_dp_preprocessing_fn

def register_dp_collate(name):
    def register_dp_collate_fn(cls):
        if name in __DP_COLLATE__:
            raise ValueError(f'Collating datapipe {name} already registered!')
        __DP_COLLATE__[name] = cls
        return cls
    return register_dp_collate_fn

def register_len_fn(name):
    def decorator(fn):
        if name in __LEN_FN__:
            raise ValueError(f'Len function {name} already registered!')
        __LEN_FN__[name] = fn
        return fn
    return decorator

def register_dp_overwrite(name):
    def register_dp_overwrite_fn(cls):
        if name in __DP_OVERWRITE__:
            raise ValueError(f'Overwrite datapipe {name} already registered!')
        __DP_OVERWRITE__[name] = cls
        return cls
    return register_dp_overwrite_fn


def create_dp_from_config(config, data_root, data_mask,
    name = '',
    chunk = False,
    drop_last = True
):

    user_dp_overwrite_key = config['data_dp_overwrite', '']
    if user_dp_overwrite_key != '' and user_dp_overwrite_key in __DP_OVERWRITE__:
        return create_dp_overwrite_from_config(config)

    pipe = (
        dp.iter.FileLister(root=data_root, masks=data_mask, recursive=False, abspath=True)
        .shuffle(buffer_size=100)
        .open_files(mode="b")
        .load_from_tar()
    )

    pipe = create_decoding_dp_from_config(config, pipe)

    pipe = (
        pipe.webdataset()
        .shuffle(buffer_size=config['buffer_size_shuffle_before_batching', 10000]) # Shuffle shards
        .sharding_filter() # Distributes across processes
    )

    pipe = create_preprocessing_dp_from_config(config, pipe)
    len_fn = get_len_fn(config)

    pipe = (
        pipe.max_token_bucketize(
            max_token_count=config['batch_size'],
            buffer_size=config['buffer_size_bucketing', 1000],
            len_fn=len_fn,
            include_padding=False
        )
        .shuffle(buffer_size=config['buffer_size_batch_shuffling', 300])
    )

    pipe = create_collating_dp_from_config(config, pipe)

    if Settings.is_gpu():
        pipe = pipe.pin_memory()
    
    if chunk:
        pipe = pipe.batch(batch_size=config['update_freq'], drop_last=drop_last)
    
    if name != '':
        name = ' ' + name

    my_print(f'Created datapipe{name}!')

    return pipe

def create_decoding_dp_from_config(config, source_dp):
    if config['data_decoding'] in __DP_DECODING__:
        pipe = __DP_DECODING__[config['data_decoding']].create_from_config(config, source_dp)
        return pipe
    else:
        raise ValueError(f'Error! Unrecognized decoding datapipe {config["data_decoding"]}!')

def create_preprocessing_dp_from_config(config, source_dp):
    if config['data_preprocess'] in __DP_PREPROCESSING__:
        pipe = __DP_PREPROCESSING__[config['data_preprocess']].create_from_config(config, source_dp)
        return pipe
    else:
        raise ValueError(f'Error! Unrecognized preprocessing datapipe {config["data_preprocess"]}!')

def create_collating_dp_from_config(config, source_dp):
    if config['data_collate'] in __DP_COLLATE__:
        pipe = __DP_COLLATE__[config['data_collate']].create_from_config(config, source_dp)
        return pipe
    else:
        raise ValueError(f'Error! Unrecognized collating datapipe {config["data_collate"]}!')

def create_dp_overwrite_from_config(config):
    user_dp_overwrite_key = config['data_dp_overwrite']
    datapipe = __DP_OVERWRITE__[user_dp_overwrite_key].create_from_config(config)
    my_print('Overwrote datapipe!')
    return datapipe


def get_all_dp_decoding_keys():
    return list(__DP_DECODING__.keys())

def get_all_dp_preprocessing_keys():
    return list(__DP_PREPROCESSING__.keys())

def get_all_len_fn_keys():
    return list(__LEN_FN__.keys())

def get_all_dp_collate_keys():
    return list(__DP_COLLATE__.keys())

def get_all_dp_overwrite_keys():
    return list(__DP_OVERWRITE__.keys())

def get_len_fn(config):
    if config['data_len_fn'] in __LEN_FN__:
        return __LEN_FN__[config['data_len_fn']]
    else:
        raise ValueError(f'Error! Unrecognized length function {config["data_len_fn"]}!')