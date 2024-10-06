import yaml
from enum import Enum
from typing import Union
from dataclasses import dataclass

from deept.utils.debug import my_print


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

        import deept.data.dataset as dt_dataset
        import deept.data.dataloader as dt_dataloader
        import deept.data.postprocessing as dt_postprocess

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['data']['dataloader'] = ConfigSpec(
            description = (f'Here you need to specify the key of your own dataloader'
                f'that you need to register with @register_dataloader(YOUR_NAME). '
                f'It is expected to have a static member function create_from_config(config, dataset, shuffle, is_train, num_workers). '
                f'that returns an object of type torch.utils.data.Dataloader.'),
            required = True,
            accepted_values = dt_dataloader.get_all_dataloader_keys()
        )

        __CONFIG_DESC__['data']['train_dataset'] = ConfigSpec(
            description = """Here you need to specify the key of your own dataset
            that you need to register with @register_dataset(YOUR_NAME).""",
            required = True,
            accepted_values = dt_dataset.get_all_dataset_keys()
        )

        __CONFIG_DESC__['data']['dev_dataset'] = ConfigSpec(
            description = """Here you need to specify the key of your own dataset
            that you need to register with @register_dataset(YOUR_NAME).""",
            required = True,
            accepted_values = dt_dataset.get_all_dataset_keys()
        )

        __CONFIG_DESC__['data']['batch_size'] = ConfigSpec(
            description = """Specify the batch_size for training here. For search we will use 'batch_size_search'.
                The batch_size is given in whatever quantity you return within length function of 'data_len_fn'.""",
            required = True,
            accepted_values = CommonAcceptedInputs.NONE_NEGATIVE_INT
        )

        __CONFIG_DESC__['data']['batch_size_search'] = ConfigSpec(
            description = """During search, the batch size specifies the number of samples, which will be sorted 
                according to the length given by 'data_len_fn'. If you would like to consider a different sorting behavior
                for search, you can implement a switch with Settings.is_training() in your length function.""",
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

        __CONFIG_DESC__['data']['buffer_sort_search'] = ConfigSpec(
            description = """Those many batches are buffered and sorted in search. 
                A higher value leads to better sorting but more RAM usage.""",
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

    @staticmethod
    def create_deept_config_description_model():

        import deept.components.model as dt_model
        import deept.components.scores as dt_scores
        import deept.components.optimizer as dt_optim
        import deept.components.lr_scheduler as dt_lrschedule

        __CONFIG_DESC__ = DeepTConfigDescription.__CONFIG_DESC__

        __CONFIG_DESC__['model']['criterions'] = ConfigSpec(
            description = (
                f'A list of criterions. A criterion is a dictionary with '
                f'two keys "score_type" and "input_keys".'
            ),
            required = True
        )

        __CONFIG_DESC__['model']['score_type'] = ConfigSpec(
            description = (f'The score identifier as specified in @register_score(score_id).'),
            required = True,
            accepted_values = dt_scores.get_all_score_keys()
        )

        __CONFIG_DESC__['model']['model'] = ConfigSpec(
            description = """The key of the model.""",
            required = True,
            accepted_values = dt_model.get_all_model_keys()
        )

        __CONFIG_DESC__['model']['optimizers'] = ConfigSpec(
            description = """TODO.""",
            required = True
        )
        
        __CONFIG_DESC__['model']['optim_type'] = ConfigSpec(
            description = """The key of an optimizer as supplied by @register_optimizer(id).""",
            required = True,
            accepted_values = dt_optim.get_all_optimizer_keys()
        )

        __CONFIG_DESC__['model']['lr_type'] = ConfigSpec(
            description = """The key of an lr_scheduler as supplied by @register_lr_scheduler(id).""",
            required = True,
            accepted_values = dt_lrschedule.get_all_lr_scheduler_keys()
        )

    @staticmethod
    def create_deept_config_description_checkpointing():
        pass

    @staticmethod
    def create_deept_config_description_search():

        import deept.search.search_algorithm as dt_search

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

    def asdict(self):
        return self.config

    def print_config(self):

        max_key_length = max([len(k) for k in self.config.keys()])
        for key in self.config:
            my_print(key.ljust(max_key_length, '-'), str(self.config[key]).ljust(100, '-'))

    def assert_has_key(self, key):
        if not self.has_key(key):
            error_msg = f'Config is missing key "{key}".'
            if DeepTConfigDescription.has_key(key):
                error_msg += (f'\nDescription: {DeepTConfigDescription[key].description}.')
            raise ValueError(error_msg)
     
    def has_key(self, key):
        return key in self.config.keys()

    def __getitem__(self, key):
        if isinstance(key, tuple):
            assert len(key) == 2
            default = key[1]
            key = key[0]
            return self.__get_item_with_default(key, default)
        else:
            return self.__get_item(key)
    
    def __get_item(self, key):
        self.assert_has_key(key)
        item = self.__parse_item(self.config[key])
        self.assert_check_item(key, item)
        return item

    def __get_item_with_default(self, key, default):
        if self.has_key(key):    
            item = self.__parse_item(self.config[key])
            self.assert_check_item(key, item)
            return item
        else:
            return self.__parse_item(default)

    def __parse_item(self, item):
        if isinstance(item, dict):
            return Config(item)
        else:
            return item

    def assert_check_item(self, key, item):
        if not DeepTConfigDescription.has_key(key):
            return

        if (not isinstance(item, str) 
            and not isinstance(item, int)
            and not isinstance(item, float)
            and not isinstance(item, bool)
        ):
            return
        
        if  (DeepTConfigDescription[key].accepted_values is None or
            not isinstance(DeepTConfigDescription[key].accepted_values, list)):
            return

        if not item in DeepTConfigDescription[key].accepted_values:
            error_msg = f'Config has invalid input for "{key}".'
            error_msg += (f'\nDescription: {DeepTConfigDescription[key].description}.'
                f'\n Accepted Values: {DeepTConfigDescription[key].accepted_values}'
                f'\n Got: {item}')
            raise ValueError(error_msg)

    def __setitem__(self, key, value):
        self.config[key] = value
    
    def update(self, dict, prefix=''):
        for k, v in dict.items():
            self.config[f'{prefix}{k}'] = v

    def items(self):
        return self.config.items()

    def keys(self):
        return self.config.keys()