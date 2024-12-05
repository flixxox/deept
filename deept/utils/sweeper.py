
import random
from os.path import join
from copy import deepcopy

from deept.utils.debug import my_print
from deept.utils.globals import Settings
from deept.utils.log import (
    value_to_str,
    write_to_file,
    round_if_float,
    write_dict_to_yaml
)

__SWEEPER_DICT__ = {}

def register_sweeper(name):
    def register_sweeper_fn(cls):
        if name in __SWEEPER_DICT__:
            raise ValueError(f'Sweeper {name} already registered!')
        __SWEEPER_DICT__[name] = cls
        return cls

    return register_sweeper_fn

def create_sweeper_from_config(config, function, function_args):
    sweeper = config['sweep_configuration']['method']
    if sweeper in __SWEEPER_DICT__:
        return __SWEEPER_DICT__[sweeper].create_from_config(config, function, function_args)
    else:
        raise ValueError(f'Error! Unrecognized sweeper {sweeper}!')


class Sweeper:

    def __init__(self,
        normal_config,
        function,
        function_args,
        best_indicator,
        best_goal,
        **kwargs
    ):
        
        self.normal_config = normal_config
        self.function = function
        self.function_args = function_args
        self.best_indicator = best_indicator
        self.best_goal = best_goal

        for k, v in kwargs.items():
            setattr(self, k, v)

        assert self.function is not None
        assert self.function_args is not None
        assert self.normal_config is not None
        assert self.best_indicator is not None
        assert self.best_goal is not None
        assert self.max_count is not None
        assert self.sweep_parameters is not None
        assert self.output_folder_root is not None

        if self.constraints is not None and not isinstance(self.constraints, list):
            self.constraints = list(self.constraints)
            for constraint in self.constraints:
                if not isinstance(constraint, str):
                    raise ValueError(f'Constraint must be a string. Got: {constraint}!')
            
        if self.best_goal == 'max':
            self.reduce_fn = max
        elif self.best_goal == 'min':
            self.reduce_fn = min
        else:
            raise ValueError(f'Did not regonize the goal of the best score. Got: {self.best_goal}!')

        self.parse_sweep_parameters()

        self.sorted_parameter_names = [k for k in self.parsed_sweep_parameters.keys()]
        self.sorted_parameter_names.sort(key=len, reverse=True)

        self.results = {}
        self.performance_sorted_configs = []
    
    def parse_sweep_parameters(self):
        self.num_combinations = 1
        self.parsed_sweep_parameters = {}
        for name, sweep_config in self.sweep_parameters.items():
            if 'values' in sweep_config:
                self.parsed_sweep_parameters[name] = sweep_config['values']
            else:
                raise ValueError('At the moment the sweep parameter config requires the "values" field!')
            self.num_combinations *= len(self.parsed_sweep_parameters[name])

    def sweep(self):
        for i in range(min(self.max_count, self.num_combinations)):
            sweep_config = self.get_config(i)
            sweep_config_as_string = self.get_sweep_config_as_string(sweep_config)

            my_print('~~~~~~ SWEEPER ~~~~~~')

            if not self.fulfills_constraints(sweep_config):
                my_print(f'Sweep config {sweep_config_as_string} does not fulfill constraints.')
                my_print('Skip!')
            else:
                my_print(f'Start sweep with: {sweep_config_as_string}')
                self.call_sweep_fn_and_log(sweep_config, sweep_config_as_string)
        
        self.log_all_best_summaries()    

    def call_sweep_fn_and_log(self, sweep_config, sweep_config_as_string):
        config = self.merge_normal_and_sweep_config(sweep_config, sweep_config_as_string)
        result = self.function(config, *self.function_args)
        self.results[sweep_config_as_string] = result
        self.update_performance_sorted_list(result, sweep_config_as_string)
        
    def get_sweep_config_as_string(self, sweep_config):
        as_string = ''
        for k, v in sweep_config.items():
            v = value_to_str(v, no_precise=True)
            as_string = f'{as_string}__{k}_{v}'
        as_string = as_string[2:]
        return as_string 

    def fulfills_constraints(self, sweep_config):
        if self.constraints is None:
            return True
        for c in self.constraints:
            for p in self.sorted_parameter_names:
                c = c.replace(p, str(sweep_config[p]))
            r = eval(c)
            if not r:
                return False
        return True

    def merge_normal_and_sweep_config(self, sweep_config, sweep_config_as_string):
        config = deepcopy(self.normal_config)

        output_folder_for_run = ''
        for k in config['sweep_configuration']['parameters'].keys():
            v = sweep_config[k]
            config[k] = v
            v = value_to_str(v, no_precise=True)
            output_folder_for_run = f'{output_folder_for_run}__{k}_{v}'
        output_folder_for_run = output_folder_for_run[2:]
        
        config['output_folder_root'] = config['output_folder']
        config['output_folder'] = join(config['output_folder'], output_folder_for_run)
        config['experiment_name'] =  f'{config["experiment_name"]}-{sweep_config_as_string}'

        return config

    def update_performance_sorted_list(self, result, sweep_config_as_string):
        this_best = result['eval'].get_best_value(self.best_indicator, self.reduce_fn)
        self.performance_sorted_configs.append((this_best, sweep_config_as_string))
        self.performance_sorted_configs.sort(key=lambda x: x[0], reverse=True)
        write_to_file('output_dir_root', 'performance_sorted_sweeps', '~~~~ NEW SWEEP ~~~~')
        for (metric, config) in self.performance_sorted_configs:
            write_to_file('output_dir_root', 'performance_sorted_sweeps', f'{config}: {value_to_str(metric)}')

    def log_all_best_summaries(self):
        to_log = {}
        for sweep_str, summary_managers in self.results.items():
            best_ckpt_idx, best_eval_summary = summary_managers['eval'].get_summary_of_best()
            best_train_summary = summary_managers['train'].get_by_index(best_ckpt_idx)
            to_log[sweep_str] = {}
            to_log[sweep_str]['train'] = {}
            to_log[sweep_str]['eval'] = {}
            to_log[sweep_str]['best_ckpt'] = best_ckpt_idx+1
            
            for k, v in best_train_summary.items():
                to_log[sweep_str]['train'][k] = round_if_float(v)
            
            for k, v in best_eval_summary.items():
                to_log[sweep_str]['eval'][k] = round_if_float(v)
        
        output_dir = Settings.get_dir('output_dir_root')
        output_dir = join(output_dir, f'sweep_summary.yaml')
        write_dict_to_yaml(output_dir, to_log)


@register_sweeper('grid')
class GridSweeper(Sweeper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_parsed_parameters()
    
    def update_parsed_parameters(self):
        i = 0
        updated_sweep_parameters = {}
        for k, v in self.parsed_sweep_parameters.items():
            updated_sweep_parameters[i] = {}
            updated_sweep_parameters[i]['name'] = k
            updated_sweep_parameters[i]['values'] = v
            i += 1
        
        for i in range(len(updated_sweep_parameters)):
            c = 1
            for j in range(i+1, len(updated_sweep_parameters)):
                c *= len(updated_sweep_parameters[j]['values'])
            updated_sweep_parameters[i]['div'] = c
        
        self.parsed_sweep_parameters = updated_sweep_parameters

    @staticmethod
    def create_from_config(config, function, function_args):
        return GridSweeper(
            config, function, function_args,
            config['best_checkpoint_indicator'],
            config['best_checkpoint_indicator_goal'],
            max_count = config['sweep_configuration']['count'],
            constraints = config['sweep_configuration']['constraints', None],
            sweep_parameters = config['sweep_configuration']['parameters'],
            output_folder_root = config['output_folder'],
        )
    
    def get_config(self, iteration):
        idx = iteration
        sweep_config = {}
        for k, v in self.parsed_sweep_parameters.items():

            div = self.parsed_sweep_parameters[k]['div']

            i = idx // div
            idx = idx % div

            name = self.parsed_sweep_parameters[k]['name']
            value = self.parsed_sweep_parameters[k]['values'][i]

            sweep_config[name] = value
        
        return sweep_config


@register_sweeper('random')
class RandomSweeper(GridSweeper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_from = list(range(self.num_combinations))

    @staticmethod
    def create_from_config(config, function, function_args):
        return RandomSweeper(
            config, function, function_args,
            config['best_checkpoint_indicator'],
            config['best_checkpoint_indicator_goal'],
            max_count = config['sweep_configuration']['count'],
            constraints = config['sweep_configuration']['constraints', None],
            sweep_parameters = config['sweep_configuration']['parameters'],
            output_folder_root = config['output_folder']
        )

    def get_config(self, iteration):
        idx = random.choice(self.draw_from)
        self.draw_from.remove(idx)
        sweep_config = super().get_config(idx)
        return sweep_config