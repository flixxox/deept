
import random

from deept.utils.debug import my_print
from deept.utils.log import value_to_str

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

    def __init__(self, function, function_args, **kwargs):

        self.function = function
        self.function_args = function_args

        for k, v in kwargs.items():
            setattr(self, k, v)

        assert self.metric is not None and isinstance(self.metric, str)
        assert self.maximize is not None and isinstance(self.maximize, bool)
        assert self.function is not None
        assert self.function_args is not None
        assert self.max_count is not None
        assert self.sweep_parameters is not None
        assert self.output_folder_root is not None

        if self.constraints is not None and not isinstance(self.constraints, list):
            self.constraints = list(self.constraints)
            for constraint in self.constraints:
                if not isinstance(constraint, str):
                    raise ValueError(f'Constraint must be a string. Got: {constraint}!')
            
        if self.maximize:
            self.reduce_fn = max
        else:
            self.reduce_fn = min

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
                result = self.function(sweep_config=sweep_config)
                self.results[sweep_config_as_string] = result
                self.log_sweep_result(result, sweep_config_as_string)
                self.log_average_so_far(sweep_config_as_string)
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

    def log_sweep_result(self, result, sweep_config_as_string):
        write_to_file(self.output_folder_root, 'sweep_summary', f' ~~~ {sweep_config_as_string} ~~~ ') 
        log_summary(self.output_folder_root, 'sweep_summary', result,
            best_ind_test='test_acc',
            best_ind_train='train_acc',
            log_train=True
        )

    def log_average_so_far(self, sweep_config_as_string):
        test_metrics = [self.reduce_fn(result['test'][self.metric]) for result in self.results.values()]
        avg = sum(test_metrics) / len(test_metrics)
        var = sum([(avg-x)**2 for x in test_metrics])/len(test_metrics)
        write_to_file(self.output_folder_root, 'sweep_summary', f'Avg {self.metric} so far: {avg:4.2f}')
        write_to_file(self.output_folder_root, 'sweep_summary', f'Var {self.metric} so far: {var:4.2f}')
        my_print(f'Avg {self.metric} so far: {avg:4.2f}')
        my_print(f'Var {self.metric} so far: {var:4.2f}')

    def update_performance_sorted_list(self, result, sweep_config_as_string):
        this_best = self.reduce_fn(result['test'][self.metric])
        self.performance_sorted_configs.append((this_best, sweep_config_as_string))
        self.performance_sorted_configs.sort(key=lambda x: x[0], reverse=True)
        write_to_file(self.output_folder_root, 'performance_sorted_configs', '~~~~ NEW SWEEP ~~~~')
        for (metric, config) in self.performance_sorted_configs:
            write_to_file(self.output_folder_root, 'performance_sorted_configs', f'{config}: {metric:4.2f}')


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
            function, function_args,
            metric = config['sweep_configuration']['metric']['name', 'test_acc'],
            maximize = config['sweep_configuration']['metric']['maximize', True],
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
            function, function_args,
            metric = config['sweep_configuration']['metric']['name', 'test_acc'],
            maximize = config['sweep_configuration']['metric']['maximize', True],
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