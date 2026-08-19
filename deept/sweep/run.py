from enum import Enum

from deept.utils.log import (
    value_to_str
)


class SweepRun:

    def __init__(self,
        config
    ):
        self.config = config
        self.ident = self.config_as_string()

        self.run_id = None
        self.result = None
        self.__has_result = False
        self.output_folder = None
        self.resume_output_folder = None
    
    def config_as_string(self):
        as_string = ''
        for k, v in sorted(self.config.items(), key=lambda item: item[0]):
            v = value_to_str(v, no_precise=False)
            as_string = f'{as_string}__{k}_{v}'
        as_string = as_string[2:]
        return as_string

    def has_result(self):
        return self.__has_result
    
    def set_result(self, result):
        self.result = result
        self.__has_result = True

    def get_result(self):
        assert self.has_result()
        return self.result

    def get_result_keys(self):
        keys = []
        for k in self.result.keys():
            keys.append(f'{k}')
            keys.append(f'{k}_std')
        return keys

    def get_result_keys_values(self):
        keys = []
        values = []
        for k, v in self.result.items():
            keys.append(f'{k}')
            keys.append(f'{k}_std')
            values.append(v[0])
            values.append(v[1])
        return keys, values

    def update_config(self, new_config):
        for k, v in new_config.items():
            self.config[k] = v
        self.ident = self.config_as_string()