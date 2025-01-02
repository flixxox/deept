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
    
    def config_as_string(self):
        as_string = ''
        for k, v in self.config.items():
            v = value_to_str(v, no_precise=True)
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