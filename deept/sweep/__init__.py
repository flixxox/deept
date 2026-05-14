from cool_config import CoolConfig

from deept.sweep.sweeper import SearchSweeper, ComparativeSweeper
from deept.sweep.strategies import create_sweep_strategy_from_config
from deept.sweep.parsing import parse_repeat_for_value, parse_sweep_parameters


def create_sweeper_from_config(config, sweep_fn, sweep_fn_args):
    shared_kwargs = build_shared_sweeper_kwargs(config, sweep_fn, sweep_fn_args)

    if isinstance(shared_kwargs['sweep_parameters'], list):
        return create_comparative_sweeper_from_config(config, shared_kwargs)
    elif isinstance(shared_kwargs['sweep_parameters'], CoolConfig):
        return create_search_sweeper_from_config(config, shared_kwargs)
    else:
        raise ValueError(
            f'Sweep config "parameters" must be either a list or a dict!'
            f'Note that depending on what it is the behavior of the sweeper differs.'
        )

def build_shared_sweeper_kwargs(config, sweep_fn, sweep_fn_args):
    do_multi_sweep, multi_sweep_kwargs = parse_multi_sweep_config(config)

    repeat_for = config['sweep_configuration/repeat_for', None]
    if repeat_for is not None:
        repeat_for = parse_repeat_for_value(config, repeat_for)
    else:
        repeat_for = []

    shared_kwargs = dict(
        normal_config=config,
        function=sweep_fn,
        function_args=sweep_fn_args,
        best_indicator=config['best_checkpoint_indicator'],
        best_goal=config['best_checkpoint_indicator_goal'],
        sweep_name=config[
            'sweep_configuration/sweep_name',
            config['sweep_configuration/multi_sweep/sweep_name', None]
        ],
        output_folder_root=config['output_folder'],
        constraints=config['sweep_configuration/constraints', None],
        sweep_parameters=config['sweep_configuration/parameters'],
        run_dry=config['run_dry', False],
        repeat_for_configs=repeat_for,
        do_multi_sweep=do_multi_sweep,
        **multi_sweep_kwargs
    )

    if shared_kwargs['sweep_name'] is None:
        raise ValueError('Missing sweep_name in sweep config!')

    return shared_kwargs

def parse_multi_sweep_config(config):
    do_multi_sweep = config['sweep_configuration/activate_multi_sweep', False]
    multi_sweep_kwargs = {
        'sweep_folder_root': None,
        'hash_config': None,
        'cleanup_after': None,
        'remove_from_hash': None,
    }

    if do_multi_sweep:
        multi_sweep_kwargs['sweep_folder_root'] = config['sweep_configuration/multi_sweep/sweep_folder_root']
        multi_sweep_kwargs['hash_config'] = config['sweep_configuration/multi_sweep/hash_config', True]
        multi_sweep_kwargs['cleanup_after'] = config['sweep_configuration/multi_sweep/cleanup_after', 86]
        multi_sweep_kwargs['remove_from_hash'] = config['sweep_configuration/multi_sweep/remove_from_hash', []]

    return do_multi_sweep, multi_sweep_kwargs

def create_comparative_sweeper_from_config(config, shared_kwargs):
    import random
    from copy import deepcopy

    configs_to_compare = deepcopy(shared_kwargs['sweep_parameters'])

    if config['shuffle_parameters_randomly', True]:
        random.Random(0).shuffle(configs_to_compare)

    comparative_kwargs = dict(
        configs_to_compare=configs_to_compare
    )

    return ComparativeSweeper(
        comparative_kwargs,
        **shared_kwargs
    )

def create_search_sweeper_from_config(config, shared_kwargs):
    param_options, num_combinations = parse_sweep_parameters(
        shared_kwargs['sweep_parameters']
    )

    sweep_strat = create_sweep_strategy_from_config(config)

    search_kwargs = dict(
        sweep_strat=sweep_strat,
        param_options=param_options,
        num_combinations=num_combinations,
        max_count=config['sweep_configuration/count'],
    )

    return SearchSweeper(
        search_kwargs,
        **shared_kwargs
    )
