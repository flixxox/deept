from cool_config import CoolConfig

from deept.sweep.sweeper import Sweeper
from deept.sweep.strategies import create_sweep_strategy_from_config

def create_sweeper_from_config(config, sweep_fn, sweep_fn_args):
    sweep_strat = create_sweep_strategy_from_config(config)
    
    do_multi_sweep = config['sweep_configuration/activate_multi_sweep', False]
    sweep_name = None
    hash_config = None
    cleanup_after = None
    remove_from_hash = None
    sweep_folder_root = None
    if do_multi_sweep:
        config['sweep_configuration'].assert_has_key('multi_sweep')
        hash_config = config['sweep_configuration/multi_sweep/hash_config', True]
        cleanup_after = config['sweep_configuration/multi_sweep/cleanup_after', 12]
        sweep_folder_root = config['sweep_configuration/multi_sweep/sweep_folder_root']
        remove_from_hash = config['sweep_configuration/multi_sweep/remove_from_hash', []]

    sweep_name = config['sweep_configuration/multi_sweep/sweep_name', None]
    config['experiment_name'] = sweep_name

    repeat_for = config['sweep_configuration/repeat_for', None]
    if repeat_for is not None:
        repeat_for = parse_repeat_for_value(config, repeat_for)
    else:
        repeat_for = []

    sweeper = Sweeper(
        config, sweep_strat, sweep_fn, sweep_fn_args,
        config['best_checkpoint_indicator'],
        config['best_checkpoint_indicator_goal'],
        experiment_name = config['experiment_name'],
        output_folder_root = config['output_folder'],
        max_count = config['sweep_configuration/count'],
        constraints = config['sweep_configuration/constraints', None],
        sweep_parameters = config['sweep_configuration/parameters', {}],
        repeat_for_configs = repeat_for,
        do_multi_sweep = do_multi_sweep,
        sweep_folder_root = sweep_folder_root,
        sweep_name = sweep_name,
        hash_config = hash_config,
        cleanup_after = cleanup_after,
        remove_from_hash = remove_from_hash
    )

    return sweeper

def parse_repeat_for_value(config, repeat_for):
    parsed_repeat_for = []

    if isinstance(repeat_for, CoolConfig):
        if len(repeat_for.keys()) > 1:
            raise ValueError(
                f'If repeat_for is given as a dict, there is only one parameter allowed. Got {len(repeat_for.keys())}: {repeat_for}!'
            )
        
        k, v = next(repeat_for.items())

        if not k in config.keys():
            raise ValueError(
                f'There is a repeat_for parameter not present in the config by default! To prevent unseen errors this throws. Got {k}!'
            )
        
        if isinstance(v, list):
            for e in v:
                parsed_repeat_for.append({k: e})
        else:
            parsed_repeat_for.append({k: v})

    elif isinstance(repeat_for, list):
        for repeat_config in repeat_for:
            if not isinstance(repeat_config, CoolConfig):
                raise ValueError(
                    f'If repeat_for contains a list, every item is expected to be a dictionary. Got: {repeat_for}!'
                )

            for k in repeat_config.keys():
                if not k in config.keys():
                    raise ValueError(
                        f'There is a repeat_for parameter not present in the config by default! To prevent unseen errors this throws. Param "{k}"!'
                    )

            parsed_repeat_for.append(repeat_config)
    else:
        raise ValueError(f'Error! repeat_for must be provided as a dict or list! Got: {repeat_for}!')

    # Repeat for has structure:[{k:v, k:v}, {k:v, k:v}, {k:v, k:v}]
    # The parameter is repeated over these configs

    return parsed_repeat_for