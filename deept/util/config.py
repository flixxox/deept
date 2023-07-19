import yaml
from enum import Enum
from typing import Union
from dataclasses import dataclass

import deept.util.data as dt_data
from deept.util.debug import my_print


class CommonAcceptedInputs(str, Enum):

    NONE_NEGATIVE_INT = '[0-9]+'
    FILEPATH = 'A filepath.'
    UNKNOWN = 'Not known.'


@dataclass
class ConfigSpec:
    required:           bool
    description:        str
    accepted_values:    Union[list, str] = CommonAcceptedInputs.UNKNOWN


class DeepTConfigDescription(object):
    
    __CONFIG_DESC__ = {}

    @staticmethod
    def create_deept_config_description():

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['number_of_gpus'] = ConfigSpec(
            description = """The parameter specifies how many GPUs are used for training/search.
                If 0 we use cpu, if at least 1 we expect CUDA to be installed.
                If >1 we will use DDP and distribute gradients to implement multi-gpu training. The parameter can be overwritten
                via the CLI during training. In search we limit it to a maximum of 1.""",
            required = True,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['model_input'] = ConfigSpec(
            description = """Every model that shall be trained with deept should have an attribute "model_input".
                It is expected to be a list of strings. Each string refers to a data key.
                DeepT will take the tensors corresponding to those keys and feed them to the __call__/forward function of the model.""",
            required = True,
            accepted_values = CommonAcceptedInputs.UNKNOWN
        )

        __CONFIG_DESC__['optimizer_input'] = ConfigSpec(
            description = """Todo""",
            required = True,
        )

        __CONFIG_DESC__['data_train_root'] = ConfigSpec(
            description = """The directory of the train webdataset tar files.""",
            required = False,
            accepted_values = CommonAcceptedInputs.FILEPATH
        )

        __CONFIG_DESC__['data_train_mask'] = ConfigSpec(
            description = """A mask that selects files within the 'data_train_root' directory as used for training.
            Make sure that it specifies .tar files.""",
            required = False,
            accepted_values = 'For example: "*.tar", "train*.tar", "train.tar".'
        )

        __CONFIG_DESC__['data_dev_root'] = ConfigSpec(
            description = """The directory of the dev webdataset tar files.""",
            required = False,
            accepted_values = CommonAcceptedInputs.FILEPATH
        )

        __CONFIG_DESC__['data_dev_mask'] = ConfigSpec(
            description = """A mask that selects files within the 'data_dev_root' directory as used for evaluation.
            Make sure that it specifies .tar files.""",
            required = False,
            accepted_values = 'For example: "*.tar", "dev*.tar", "dev.tar".'
        )

        __CONFIG_DESC__['data_decoding'] = ConfigSpec(
            description = """Here you can specify the datapipe that decodes the binary and uncompressed datastream from the webdataset.
            You can register your own decoding datapipe with '@register_dp_decoding(YOUR_KEY)'.""",
            required = False,
            accepted_values = dt_data.get_all_dp_decoding_keys()
        )

        __CONFIG_DESC__['data_preprocess'] = ConfigSpec(
            description = """Here you specify the key of the preprocessing data pipeline. It takes as input the raw data (already decoded) and
            can perform some arbitrary preprocessing. For example, in machine translation this refers to appending special symbols and tokenizing.
            Note that the returned items should not be padded or coverted to tensors yet.
            With '@register_dp_preprocessing(YOUR_KEY)' you can register your own preprocessing pipeline.""",
            required = False,
            accepted_values = dt_data.get_all_dp_preprocessing_keys()
        )

        __CONFIG_DESC__['data_collate'] = ConfigSpec(
            description = """Todo""",
            required = False,
        )

        __CONFIG_DESC__['data_len_fn'] = ConfigSpec(
            description = """With this parameter you specify the size of one data sample used for batching. 
            The len_fn takes as input one data sample and returns an integer that represents the size of that sample.
            Whatever you return here will be the quantity of 'batch_size'.
            You can register your own length function with '@register_len_fn(YOUR_KEY)'.""",
            required = False,
            accepted_values = dt_data.get_all_len_fn_keys()
        )

        __CONFIG_DESC__['data_dp_overwrite'] = ConfigSpec(
            description = """Normally we use a generic webdataset datapipe and insert user-specific datapipes along the way.
            If you want to only use your own datapipe register it with "@register_dp_overwrite(YOUR_NAME)" and implement the static function
            create_from_config(config). 'data_dp_overwrite' refers to the datapipe key, i.e. 'YOUR_NAME'.
            If 'data_dp_overwrite' is not specified, you must specify the parameters: "data_decoding", "data_preprocess" and "data_collate".""",
            required = False,
            accepted_values = dt_data.get_all_dp_overwrite_keys()
        )

    @staticmethod
    def has_key(key):
        return key in DeepTConfigDescription.__CONFIG_DESC__

    @staticmethod
    def print_all_deept_arguments():
        for k, v in DeepTConfigDescription.__CONFIG_DESC__.items():
            
            if isinstance(v.accepted_values, Enum):
                accepted_values = v.accepted_values.value
            else:
                accepted_values = v.accepted_values
            
            my_print(f'NAME: {k}'.center(40, '-'))
            my_print(f'{"Required".ljust(16, " ")}: {v.required}')
            my_print(f'{"Description".ljust(16, " ")}: {v.description}')
            my_print(f'{"Accepted Values".ljust(16, " ")}: {accepted_values}')

    def __class_getitem__(cls, key):
        return DeepTConfigDescription.__CONFIG_DESC__[key]


class Config:

    def __init__(self, config):

        self.config = config

    @staticmethod
    def parse_config(args):

        with open(args['config'], 'r') as f:    
            config = yaml.safe_load(f)
        
        assert config is not None, 'Provided config seems to be empty' 

        for k, v in args.items():
            if v is not None:
                if k in config.keys():
                    my_print(f'CLI config overwrite for "{k}"!')
                config[k] = v

        return Config(config)

    def print_config(self):

        max_key_length = max([len(k) for k in self.config.keys()])
        for key in self.config:
            my_print(key.ljust(max_key_length, '-'), str(self.config[key]).ljust(100, '-'))

    def assert_has_key(self, key):
        
        if not self.has_key(key):
            error_msg = f'Config is missing key "{key}".'
            if DeepTConfigDescription.has_key(key):
                error_msg += f'\nDescription: {DeepTConfigDescription[key].description}'
            raise ValueError(error_msg)

    def has_key(self, key):
        return key in self.config.keys()

    def __getitem__(self, key):

        default = None

        if isinstance(key, tuple):
            assert len(key) == 2
            default = key[1]
            key = key[0]

        if default is None:
            self.assert_has_key(key)
            return self.config[key]
        else:
            if self.has_key(key):    
                return self.config[key]
            else:
                return default

    def __setitem__(self, key, value):
        self.config[key] = value