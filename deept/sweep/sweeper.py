import numpy as np
from os.path import join
from copy import deepcopy

from deept.sweep.run import SweepRun
from deept.utils.config import Config
from deept.utils.debug import my_print
from deept.utils.globals import Settings
from deept.sweep.database import SweepDatabase
from deept.utils.log import (
    value_to_str,
    write_to_file,
    round_if_float,
    write_dict_to_yaml
)


class Sweeper:

    def __init__(self,
        normal_config,
        sweep_strat,
        function,
        function_args,
        best_indicator,
        best_goal,
        **kwargs
    ):
        self.normal_config = normal_config
        self.sweep_strat = sweep_strat
        self.function = function
        self.function_args = function_args
        self.best_indicator = best_indicator
        self.best_goal = best_goal

        for k, v in kwargs.items():
            setattr(self, k, v)

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

        self.do_repeat_for_seeds = len(self.seeds_to_try) > 0

        self.__parse_sweep_parameters()
        self.sweep_strat.parse_sweep_parameters(self.param_options)

        self.sorted_parameter_names = [k for k in self.param_options.keys()]
        self.sorted_parameter_names.sort(key=len, reverse=True)

        if self.do_multi_sweep:
            self.__create_database()
            self.database.connect()

        self.results = {}
        self.performance_sorted_configs = []

    def __parse_sweep_parameters(self):
        self.num_combinations = 1
        self.param_options = {}
        for name, sweep_config in self.sweep_parameters.items():
            keys = sweep_config.keys()
            if 'values' in keys:
                for v in sweep_config['values']:
                    if isinstance(v, str) and '_' in v:
                        raise ValueError(
                            'Currently, "_" is a special token and not supported as part of a value.'
                        )
                self.param_options[name] = sweep_config['values']
            elif 'max' in keys and 'min' in keys and 'step' in keys:
                self.param_options[name] = self.__parse_sweep_params_given_by_range(
                    sweep_config['min'], sweep_config['max'], sweep_config['step'], sweep_config['round_to', None]
                )
            else:
                raise ValueError('Incorrect sweep param specification!')
            self.num_combinations *= len(self.param_options[name])
    
    def __parse_sweep_params_given_by_range(self, min, max, step, round_to):
        values = list(np.arange(min, max+step, step))

        if round_to is None:
            round_to = self.__autodetect_round_precision(min, max, step)
            return [round(v, round_to) for v in values]
        else:
            if round_to.lowercase() == 'no_round':
                return values
            else:
                if isinstance(round_to, int):
                    return [round(v, round_to) for v in values]
                else:
                    raise ValueError(f'Got wrong value for round_to. Got {round_to}. Expecting int or "no_round"!')

    def __autodetect_round_precision(self, smin, smax, sstep):
        def __get_decimal_digits(inp):
            if '.' in inp:
                return len(inp.split('.')[-1])
            else:
                return 0

        smin = str(smin)
        smax = str(smax)
        sstep = str(sstep)

        precision = max(0, __get_decimal_digits(smin))
        precision = max(precision, __get_decimal_digits(smax))
        precision = max(precision, __get_decimal_digits(sstep))

        return precision

    def __create_database(self):
        sweep_name = self.experiment_name
        if self.sweep_name is not None and self.sweep_name != '':
            sweep_name = self.sweep_name

        self.database = SweepDatabase(
            self.normal_config,
            self.sweep_folder_root,
            sweep_name,
            self.hash_config,
            self.cleanup_after,
            self.remove_from_hash
        )

    def sweep(self):
        for i in range(min(self.max_count, self.num_combinations)):
            run_config = self.sweep_strat.get_config(i)
            run = SweepRun(run_config)
            
            if self.is_valid(run):
                if self.do_multi_sweep:
                    self.database.mark_running(run)
                
                my_print(f'Sweeper: Running {run.ident}!')

                self.call_sweep_fn_and_log(run_config, run.ident)
                
                if self.do_multi_sweep:
                    run.set_result(self.results[run.ident])
                    self.database.mark_done(run)

        self.log_all_best_summaries()
        if self.do_multi_sweep:
            self.database.disconnect()

    def is_valid(self, run):
        if not self.fulfills_constraints(run.config):
            my_print(f'Skip {run.ident}! Constraints not met.')
            return False
        if self.do_multi_sweep and self.database.is_already_running_or_done(run):
            my_print(f'Skip {run.ident}! Already tried.')
            return False
        return True

    def call_sweep_fn_and_log(self, run_config, run_ident):
        if self.do_repeat_for_seeds:
            result = self.call_for_every_seed(run_config, run_ident)
        else:
            result = self.call_normal(run_config, run_ident)
        self.results[run_ident] = result
        self.update_performance_sorted_list(result, run_ident)

    def call_for_every_seed(self, run_config, run_ident):
        def __store_in(result, result_storage):
            for k, v in result['train'].items():
                result_storage['train'][k].append(v)

            for k, v in result['dev'].items():
                result_storage['dev'][k].append(v)

            result_storage['best_ckpt'].append(result['best_ckpt'])

            return result_storage

        def __avg_and_std(result_storage):
            result = {'train': {}, 'dev': {}}
            for k, v in result_storage['train'].items():
                result['train'][k] = float(np.mean(v))
                result['train'][f'{k}_std'] = float(np.std(v))

            for k, v in result_storage['dev'].items():
                result['dev'][k] = float(np.mean(v))
                result['dev'][f'{k}_std'] = float(np.std(v))

            result['best_ckpt'] = float(np.mean(result_storage['best_ckpt']))
            
            return result

        seed = self.seeds_to_try[0]
        run_config['seed'] = seed
        my_print(f'Sweeper: Run for seed {seed}!')
        result = self.call_normal(run_config, f'{run_ident}__seed_{seed}')

        result_storage = {'train': {}, 'dev': {}}
        for k, v in result['train'].items():
            result_storage['train'][k] = [v]
        for k, v in result['dev'].items():
            result_storage['dev'][k] = [v]
        result_storage['best_ckpt'] = [result['best_ckpt']]
        
        for seed in self.seeds_to_try[1:]:
            my_print(f'Sweeper: Run for seed {seed}!')
            run_config['seed'] = seed
            result = self.call_normal(run_config, f'{run_ident}__seed_{seed}')
            result_storage = __store_in(result, result_storage)

        result = __avg_and_std(result_storage)

        return result

    def call_normal(self, run_config, run_ident):
        config = self.merge_normal_and_run_config(run_config, run_ident)
        summary_managers = self.function(config, *self.function_args)

        best_ckpt_idx, best_dev_summary = summary_managers['dev'].get_summary_of_best()
        best_train_summary = summary_managers['train'].get_by_index(best_ckpt_idx)
        return {
            'train': best_train_summary.asdict(),
            'dev': best_dev_summary.asdict(),
            'best_ckpt': best_ckpt_idx
        }

    def fulfills_constraints(self, run_config):
        if self.constraints is None:
            return True
        for c in self.constraints:
            for p in self.sorted_parameter_names:
                c = c.replace(p, str(run_config[p]))
            r = eval(c)
            if not r:
                return False
        return True

    def merge_normal_and_run_config(self, run_config, run_ident):
        config = deepcopy(self.normal_config)

        for k, v in run_config.items():
            config[k] = v
        
        config['output_folder_root'] = config['output_folder']
        config['output_folder'] = join(config['output_folder'], run_ident)
        config['experiment_name'] =  f'{config["experiment_name"]}-{run_ident}'

        return config

    def update_performance_sorted_list(self, result, run_ident):
        this_best = result['dev'][self.best_indicator]
        self.performance_sorted_configs.append((this_best, run_ident))
        self.performance_sorted_configs.sort(key=lambda x: x[0], reverse=True)
        write_to_file('output_dir_root', 'performance_sorted_sweeps', '~~~~ NEW SWEEP ~~~~')
        for (metric, config) in self.performance_sorted_configs:
            write_to_file('output_dir_root', 'performance_sorted_sweeps', f'{config}: {value_to_str(metric)}')

    def log_all_best_summaries(self):
        to_log = {}
        for sweep_str, result in self.results.items():
            best_ckpt = result['best_ckpt']
            to_log[sweep_str] = {}
            to_log[sweep_str]['train'] = {}
            to_log[sweep_str]['dev'] = {}
            to_log[sweep_str]['best_ckpt'] = best_ckpt+1
            
            for k, v in result['train'].items():
                to_log[sweep_str]['train'][k] = round_if_float(v)
            
            for k, v in result['dev'].items():
                to_log[sweep_str]['dev'][k] = round_if_float(v)
        
        output_dir = Settings.get_dir('output_dir_root')
        output_dir = join(output_dir, f'sweep_summary.yaml')
        write_dict_to_yaml(output_dir, to_log)