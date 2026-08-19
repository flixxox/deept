import numpy as np
from os.path import join
from copy import deepcopy

from deept.sweep.run import SweepRun
from deept.utils.config import Config
from deept.utils.debug import my_print
from deept.utils.globals import Settings
from deept.sweep.database import SweepDatabase
from deept.utils.log import (
    Summary,
    value_to_str,
    write_to_file,
    round_if_float,
    write_dict_to_yaml
)


class Sweeper:

    def __init__(self, **kwargs):
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

        self.do_repeat_for = len(self.repeat_for_configs) > 0

        if self.do_multi_sweep:
            self.__create_database()
            self.database.connect()

        self.results = {}
        self.performance_sorted_configs = []

    def __create_database(self):
        self.database = SweepDatabase(
            self.normal_config,
            self.sweep_folder_root,
            self.sweep_name,
            self.hash_config,
            self.cleanup_after,
            self.remove_from_hash
        )

    def sweep(self):
        self.inner_sweep()
        self.log_all_best_summaries()
        if self.do_multi_sweep:
            self.database.disconnect()

    def maybe_run(self, run):
        action, output_folder = self.determine_run_action(run)

        if action == 'skip':
            return False

        run.output_folder = output_folder
        run.resume_output_folder = output_folder if action == 'resume' else None

        if self.do_multi_sweep:
            self.database.mark_running(run, output_folder)

        if action == 'resume':
            my_print(f'Sweeper: Resuming {run.ident} from {output_folder}!')
        else:
            my_print(f'Sweeper: Running {run.ident}!')

        self.call_sweep_fn_and_log(run)

        if self.do_multi_sweep:
            run.set_result(self.results[run.ident])
            self.database.mark_done(run)

        return True

    def determine_run_action(self, run):
        fresh_output_folder = join(self.normal_config['output_folder'], run.ident)

        if not self.do_multi_sweep:
            return 'run', fresh_output_folder

        status = self.database.get_run_status(run)

        if status is None:
            return 'run', fresh_output_folder
        elif status['status'] == 'ERROR':
            if self.restart_error_runs:
                my_print(f'Sweeper: {run.ident} is in error state! "restart_error_runs" is set, starting over fresh!')
                return 'run', fresh_output_folder
            else:
                return 'resume', status['output_folder']
        elif status['status'] == 'RUNNING' and self.force_resume_of_running_jobs:
            my_print(f'Sweeper: {run.ident} is marked RUNNING, but "force_resume_of_running_jobs" is set! Resuming anyway!')
            return 'resume', status['output_folder']
        else:
            my_print(f'Skip {run.ident}! Already tried.')
            return 'skip', None

    def call_sweep_fn_and_log(self, run):
        if self.run_dry:
            result = Summary('')
            result.update_from_key_value(
                self.best_indicator, (0.0, 0.0)
            )
        else:
            if self.do_repeat_for:
                result = self.call_for_every_repeat_for(run)
            else:
                result = self.call_normal(run)
                for k,v in result.items():
                    result.update_from_key_value(k, (v, 0., 1))
        self.results[run.ident] = result
        self.update_performance_sorted_list(result, run.ident)

    def call_for_every_repeat_for(self, run):
        def __store_in(summary, summary_storage):
            for k, v in summary.items():
                if k in summary_storage.keys():
                    summary_storage.update_from_key_value(
                        k, summary_storage.get_value(k) + [v]
                    )
                else:
                    summary_storage.update_from_key_value(
                        k, [v]
                    )
            return summary_storage

        def __avg_and_std(summary_storage):
            avg_summary = Summary('')
            for k, v in summary_storage.items():
                assert isinstance(v, list)
                avg_summary.update_from_key_value(
                    k, (
                        round(float(np.mean(v)), 2),
                        round(float(np.std(v)), 2),
                        len(v)
                    )
                )
            return avg_summary
        
        # We modify the config only temporarily
        # for that repeat_for config
        run = deepcopy(run)

        summary_storage = Summary('')
        for repeat_for_config in self.repeat_for_configs:
            my_print(f'Sweeper: Run for repeat_for_config {repeat_for_config}!')
            run.update_config(repeat_for_config)

            summary = self.call_normal(run)
            summary_storage = __store_in(summary, summary_storage)

        result = __avg_and_std(summary_storage)

        return result

    def call_normal(self, run):
        config = self.merge_normal_and_run_config(run)
        summary = self.function(config, *self.function_args)
        assert isinstance(summary, Summary)
        return summary

    def merge_normal_and_run_config(self, run):
        config = deepcopy(self.normal_config)

        for k, v in run.config.items():
            if k not in config.keys():
                raise ValueError(f'There is a parameter in the sweep config ("{k}") which is not specified in the main config!')
            config[k] = v

        config['output_folder'] = run.output_folder if run.output_folder is not None else join(config['output_folder'], run.ident)
        config['experiment_name'] =  f'{self.sweep_name}-{run.ident}'

        if run.resume_output_folder is not None:
            checkpoint_dir_to_resume_from = join(run.resume_output_folder, 'checkpoints')
            config['load_weights'] = True
            config['load_checkpoint_from'] = checkpoint_dir_to_resume_from
            for downstream_config in config['downstreams', []]:
                downstream_config['load_weights'] = True
                downstream_config['load_checkpoint_from'] = checkpoint_dir_to_resume_from

        return config

    def update_performance_sorted_list(self, result, run_ident):
        this_best = result.get_value(self.best_indicator)
        self.performance_sorted_configs.append((this_best, run_ident))
        sort_reverse = (self.best_goal == 'max')
        self.performance_sorted_configs.sort(key=lambda x: x[0], reverse=sort_reverse)
        write_to_file('output_dir_root', 'performance_sorted_sweeps', '~~~~ NEW SWEEP ~~~~')
        for (metric, config) in self.performance_sorted_configs:
            write_to_file('output_dir_root', 'performance_sorted_sweeps', f'{config}: {value_to_str(metric)}')

    def log_all_best_summaries(self):
        if Settings.has_dir('output_dir_root'):
            to_log = {}
            for sweep_str, result in self.results.items():
                to_log[sweep_str] = {}
                for k, v in result.items():
                    to_log[sweep_str][k] = round_if_float(v)
            output_dir = Settings.get_dir('output_dir_root')
            output_dir = join(output_dir, f'sweep_summary.yaml')
            write_dict_to_yaml(output_dir, to_log)
        else:
            my_print('Warning! Did not find directory "output_dir_root". Cannot log sweep summary!')

class ComparativeSweeper(Sweeper):

    def __init__(self, comparative_kwargs, **kwargs):
        super().__init__(**kwargs)
        for k, v in comparative_kwargs.items():
            setattr(self, k, v)

    def inner_sweep(self):
        for run_config in self.configs_to_compare:
            run = SweepRun(run_config)
            self.maybe_run(run)

class SearchSweeper(Sweeper):

    def __init__(self, search_kwargs, **kwargs):
        super().__init__(**kwargs)
        for k, v in search_kwargs.items():
            setattr(self, k, v)

        self.sweep_strat.parse_sweep_parameters(self.param_options)
        self.sorted_parameter_names = [k for k in self.param_options.keys()]
        self.sorted_parameter_names.sort(key=len, reverse=True)

    def inner_sweep(self):
        i_ran = 0
        i_tried = 0
        while i_ran < self.max_count and i_tried < self.num_combinations:
            run_config = self.sweep_strat.get_config()
            run = SweepRun(run_config)

            if self.fulfills_constraints(run.config):
                did_run = self.maybe_run(run)
                if did_run:
                    i_ran += 1
            else:
                my_print(f'Skip {run.ident}! Constraints not met.')

            i_tried +=1

        if i_tried >= self.num_combinations:
            my_print(f'Sweeper: Tried all {self.num_combinations} possible combinations!')
        elif i_ran >= self.max_count:
            my_print(f'Sweeper: Ran max amount {self.max_count} of requested combinations!')
    
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