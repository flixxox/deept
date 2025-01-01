from deept.sweep.sweeper import Sweeper
from deept.sweep.strategies import create_sweep_strategy_from_config

def create_sweeper_from_config(config, sweep_fn, sweep_fn_args):

    sweep_strat = create_sweep_strategy_from_config(config)

    sweeper = Sweeper(
        config, sweep_strat, sweep_fn, sweep_fn_args,
        config['best_checkpoint_indicator'],
        config['best_checkpoint_indicator_goal'],
        experiment_name = config['experiment_name'],
        output_folder_root = config['output_folder'],
        sweep_folder_root =  config['sweep_configuration/sweep_folder_root'],
        max_count = config['sweep_configuration/count'],
        constraints = config['sweep_configuration/constraints', None],
        sweep_parameters = config['sweep_configuration/parameters'],
        hash_config = config['sweep_configuration/hash_config', True],
        sweep_name = config['sweep_configuration/sweep_name', None],
        cleanup_after = config['sweep_configuration/cleanup_after_hours', 12]
    )

    return sweeper