import yaml
from enum import Enum
from typing import Union
from dataclasses import dataclass

import deept.search as dt_search
import deept.model.model as dt_model
from deept.util.debug import my_print
import deept.model.scores as dt_scores
import deept.util.datapipe as dt_datapipe
import deept.util.dataloader as dt_dataloader
import deept.util.postprocessing as dt_postprocess


class CommonAcceptedInputs(str, Enum):

    NONE_NEGATIVE_INT = '[0-9]+'
    FILEPATH = 'A filepath.'
    UNKNOWN = 'Not known.'


@dataclass
class ConfigSpec:
    required:           bool
    description:        str
    accepted_values:    Union[list, str] = CommonAcceptedInputs.UNKNOWN


class DeepTConfigDescription:
    
    __CONFIG_DESC__ = {}

    @staticmethod
    def create_deept_config_description():

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['general'] = {}
        __CONFIG_DESC__['data'] = {}
        __CONFIG_DESC__['model'] = {}
        __CONFIG_DESC__['checkpointing'] = {}
        __CONFIG_DESC__['search'] = {}

        DeepTConfigDescription.create_deept_config_description_general()
        DeepTConfigDescription.create_deept_config_description_data()
        DeepTConfigDescription.create_deept_config_description_model()
        DeepTConfigDescription.create_deept_config_description_checkpointing()
        DeepTConfigDescription.create_deept_config_description_search()

    @staticmethod
    def create_deept_config_description_general():

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['general']['number_of_gpus'] = ConfigSpec(
            description = """The parameter specifies how many GPUs are used for training/search.
                If 0 we use cpu, if at least 1 we expect CUDA to be installed.
                If >1 we will use DDP and distribute gradients to implement multi-gpu training. The parameter can be overwritten
                via the CLI during training. In search we limit it to a maximum of 1.""",
            required = True,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['general']['best_checkpoint_indicator'] = ConfigSpec(
            description = """The score key which is used to indicate the best checkpoint.
                At the moment, only scores with descending order are considered.
                The score key is a name given to a ScoreAccumulator.""",
            required = True,
            accepted_values = CommonAcceptedInputs.UNKNOWN
        )

    @staticmethod
    def create_deept_config_description_data():

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['data']['data_train_root'] = ConfigSpec(
            description = """The directory of the train webdataset tar files.""",
            required = False,
            accepted_values = CommonAcceptedInputs.FILEPATH
        )

        __CONFIG_DESC__['data']['data_train_mask'] = ConfigSpec(
            description = """A mask that selects files within the 'data_train_root' directory as used for training.
                Make sure that it specifies .tar files.""",
            required = False,
            accepted_values = 'For example: "*.tar", "train*.tar", "train.tar".'
        )

        __CONFIG_DESC__['data']['data_dev_root'] = ConfigSpec(
            description = """The directory of the dev webdataset tar files.""",
            required = False,
            accepted_values = CommonAcceptedInputs.FILEPATH
        )

        __CONFIG_DESC__['data']['data_dev_mask'] = ConfigSpec(
            description = """A mask that selects files within the 'data_dev_root' directory as used for evaluation.
                Make sure that it specifies .tar files.""",
            required = False,
            accepted_values = 'For example: "*.tar", "dev*.tar", "dev.tar".'
        )

        __CONFIG_DESC__['data']['data_decoding'] = ConfigSpec(
            description = """Here you can specify the datapipe that decodes the binary and uncompressed datastream from the webdataset.
                You can register your own decoding datapipe with '@register_dp_decoding(YOUR_KEY)'.""",
            required = False,
            accepted_values = dt_datapipe.get_all_dp_decoding_keys()
        )

        __CONFIG_DESC__['data']['data_preprocess'] = ConfigSpec(
            description = """Here you specify the key of the preprocessing data pipeline. It takes as input the raw data (already decoded) and
                can perform some arbitrary preprocessing. For example, in machine translation this refers to appending special symbols and tokenizing.
                Note that the returned items should not be padded or coverted to tensors yet.
                With '@register_dp_preprocessing(YOUR_KEY)' you can register your own preprocessing pipeline.""",
            required = False,
            accepted_values = dt_datapipe.get_all_dp_preprocessing_keys()
        )

        __CONFIG_DESC__['data']['data_collate'] = ConfigSpec(
            description = """Here you specify the data collating  pipeline. It takes as input the batched but not padded nor tensorized items as lists.
                It should map each list to the corresponding tensor.""",
            required = False,
            accepted_values = dt_datapipe.get_all_dp_collate_keys()
        )

        __CONFIG_DESC__['data']['data_len_fn'] = ConfigSpec(
            description = """With this parameter you specify the size of one data sample used for batching. 
                The len_fn takes as input one data sample and returns an integer that represents the size of that sample.
                Whatever you return here will be the quantity of 'batch_size'.
                You can register your own length function with '@register_len_fn(YOUR_KEY)'.""",
            required = False,
            accepted_values = dt_datapipe.get_all_len_fn_keys()
        )

        __CONFIG_DESC__['data']['data_dp_overwrite'] = ConfigSpec(
            description = """Normally we use a generic webdataset datapipe and insert user-specific datapipes along the way.
                If you want to only use your own datapipe register it with "@register_dp_overwrite(YOUR_NAME)" and implement the static function
                create_from_config(config). 'data_dp_overwrite' refers to the datapipe key, i.e. 'YOUR_NAME'.
                If 'data_dp_overwrite' is not specified, you must specify the parameters: "data_decoding", "data_preprocess" and "data_collate".""",
            required = False,
            accepted_values = dt_datapipe.get_all_dp_overwrite_keys()
        )

        __CONFIG_DESC__['data']['dataloader'] = ConfigSpec(
            description = """Here you can specify the key of your own dataloader that you need to register with @register_dataloader(YOUR_NAME).
                It is expected to have a static member function create_from_config(config, datapipe).""",
            required = False,
            accepted_values = dt_dataloader.get_all_dataloader_keys()
        )

        __CONFIG_DESC__['data']['batch_size'] = ConfigSpec(
            description = """Specify the batch_size for training here. For search we will use 'batch_size_search'.
                The batch_size is given in whatever quantity you return within length function of 'data_len_fn'.""",
            required = True,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['min_sample_size'] = ConfigSpec(
            description = """The smallest size a data sample should have. If a data sample is smaller it will be discarded.
                Given in the quantity returned by 'data_len_fn'.""",
            required = False,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['max_sample_size'] = ConfigSpec(
            description = """The largest size a data sample should have. If a data sample is larger it will be discarded.
                Given in the quantity returned by 'data_len_fn'.""",
            required = False,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['buffer_size_bucketing'] = ConfigSpec(
            description = """The bucketizer buffers these many samples. Within this buffer the sentences are sorted.
            To control variability of batches keep the buffer size sufficiently small. Default=1000.
            It allows to control the tradeoff between sorting and randomness. If the buffer size is small batches will be
            more random. If the buffer size is large batches are well sorted. 
            See: https://pytorch.org/data/main/generated/torchdata.datapipes.iter.MaxTokenBucketizer.html""",
            required = False,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['buffer_size_batch_shuffling'] = ConfigSpec(
            description = """The buffer size of the last shuffling operation. These many batches will be buffered and shuffled.
            Note that all those batches need to be buffered in RAM for each worker.""",
            required = False,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['dataloader_workers'] = ConfigSpec(
            description = """The number of threads per process reading data.""",
            required = False,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['postprocessing_fn'] = ConfigSpec(
            description = """This is needed for search.
                The key to the postprocessing function. The postprocessing function 
                is directly called after the search algorithm. It should have the following signature:
                some_function_name(NAME, Tensor). Every entry in the output dict of the search algorithm 
                that is a Tensor is fed into this function with its name first. This function should
                convert the Tensor (the Tensor is batched) into a list of readable values.
                If we recognize the type we will write the values to the file search_CKPT_NAME 
                line by line in order of the dataset.""",
            required = True,
            accepted_values = dt_postprocess.get_all_postprocessing_keys()
        )

        __CONFIG_DESC__['data']['corpus_size_dev'] = ConfigSpec(
            description = """This is needed for search.
                DeepT needs to know how many samples are within the corpus to write search results in order to a file.
                The reason DeepT cannot determine it, is because a filter might be applied by the user before DeepT has 
                a chance to see the data.""",
            required = True,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['corpus_size_test'] = ConfigSpec(
            description = """Analog to 'corpus_size_dev'.""",
            required = True,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

    @staticmethod
    def create_deept_config_description_model():

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['model']['criterion'] = ConfigSpec(
            description = """The key of the score that is used as the criterion.""",
            required = True,
            accepted_values = dt_scores.get_all_score_keys()
        )

        __CONFIG_DESC__['model']['model'] = ConfigSpec(
            description = """The key of the model.""",
            required = True,
            accepted_values = dt_model.get_all_model_keys()
        )

    @staticmethod
    def create_deept_config_description_checkpointing():
        pass

    @staticmethod
    def create_deept_config_description_search():

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['search']['search_algorithm'] = ConfigSpec(
            description = """The key of the search algorithm to use.""",
            required = False,
            accepted_values = dt_search.get_all_search_algorithm_keys()
        )

        __CONFIG_DESC__['search']['search_print_per_step_keys'] = ConfigSpec(
            description = """A list of data keys that should be print after each search step.
                They refer to the data keys of the datapipeline and 
                the keys from the search algorithm output.""",
            required = False,
            accepted_values = CommonAcceptedInputs.UNKNOWN
        )

    @staticmethod
    def has_key(key):
        for topic, v in DeepTConfigDescription.__CONFIG_DESC__.items():
            if key in v:
                return True
        return False

    @staticmethod
    def print_all_deept_arguments():

        def __print_topic(topic_dict):

            for k, v in topic_dict.items():
            
                if isinstance(v.accepted_values, Enum):
                    accepted_values = v.accepted_values.value
                else:
                    accepted_values = v.accepted_values
                
                my_print(f'NAME: {k}'.center(40, '-'))
                my_print(f'{"Required".ljust(16, " ")}: {v.required}')
                my_print(f'{"Description".ljust(16, " ")}: {v.description}')
                my_print(f'{"Accepted Values".ljust(16, " ")}: {accepted_values}')

        for topic, v in DeepTConfigDescription.__CONFIG_DESC__.items():
            
            my_print(f' {topic} '.center(80, '='))

            __print_topic(v)

    def __class_getitem__(cls, key):
        for topic, v in DeepTConfigDescription.__CONFIG_DESC__.items():
            if key in v:
                return v[key]
        raise KeyError(f'Error! Could not find key "{key}" in DeepTConfigDescription.')


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

        if isinstance(key, tuple):
            
            assert len(key) == 2

            default = key[1]
            key = key[0]

            if self.has_key(key):    
                return self.config[key]
            else:
                return default

        else:
            self.assert_has_key(key)
            return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value