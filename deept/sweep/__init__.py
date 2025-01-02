from deept.sweep.sweeper import Sweeper
from deept.sweep.strategies import create_sweep_strategy_from_config

def create_sweeper_from_config(config, sweep_fn, sweep_fn_args):
    sweep_strat = create_sweep_strategy_from_config(config)
    
    do_multi_sweep = config['sweep_configuration/activate_multi_sweep', False]
    sweep_name = None
    hash_config = None
    cleanup_after = None
    sweep_folder_root = None
    if do_multi_sweep:
        config['sweep_configuration'].assert_has_key('multi_sweep')
        sweep_name = config['sweep_configuration/multi_sweep/sweep_name', None]
        hash_config = config['sweep_configuration/multi_sweep/hash_config', True]
        cleanup_after = config['sweep_configuration/multi_sweep/cleanup_after', 12]
        sweep_folder_root = config['sweep_configuration/multi_sweep/sweep_folder_root']
        remove_from_hash = config['sweep_configuration/multi_sweep/remove_from_hash', []]

    sweeper = Sweeper(
        config, sweep_strat, sweep_fn, sweep_fn_args,
        config['best_checkpoint_indicator'],
        config['best_checkpoint_indicator_goal'],
        experiment_name = config['experiment_name'],
        output_folder_root = config['output_folder'],
        max_count = config['sweep_configuration/count'],
        constraints = config['sweep_configuration/constraints', None],
        sweep_parameters = config['sweep_configuration/parameters'],
        do_multi_sweep = do_multi_sweep,
        sweep_folder_root = sweep_folder_root,
        sweep_name = sweep_name,
        hash_config = hash_config,
        cleanup_after = cleanup_after,
        remove_from_hash = remove_from_hash
    )

    return sweeper