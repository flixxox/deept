import numpy as np

from cool_config import CoolConfig


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

def parse_sweep_parameters(sweep_parameters):
    num_combinations = 1
    param_options = {}
    for name, sweep_config in sweep_parameters.items():
        keys = sweep_config.keys()
        if 'values' in keys:
            for v in sweep_config['values']:
                if isinstance(v, str) and '_' in v:
                    raise ValueError(
                        'Currently, "_" is a special token and not supported as part of a value.'
                    )
            param_options[name] = sweep_config['values']
        elif 'max' in keys and 'min' in keys and 'step' in keys:
            param_options[name] = parse_sweep_params_given_by_range(
                sweep_config['min'], sweep_config['max'], sweep_config['step'], sweep_config['round_to', None]
            )
        else:
            raise ValueError('Incorrect sweep param specification!')
        num_combinations *= len(param_options[name])

    return param_options, num_combinations
    
def parse_sweep_params_given_by_range(min, max, step, round_to):
    values = list(np.arange(min, max+step, step))

    if round_to is None:
        round_to = autodetect_round_precision(min, max, step)
        return [round(v, round_to) for v in values]
    else:
        if round_to.lower() == 'no_round':
            return values
        else:
            if isinstance(round_to, int):
                return [round(v, round_to) for v in values]
            else:
                raise ValueError(f'Got wrong value for round_to. Got {round_to}. Expecting int or "no_round"!')

def autodetect_round_precision(smin, smax, sstep):
    def __get_decimal_digits(inp):
        if inp.startswith('1e-'):
            return int(inp[3:])
        elif '.' in inp:
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