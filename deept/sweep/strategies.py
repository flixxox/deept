import random

__SWEEP_STRATEGY_DICT__ = {}

def register_sweep_strategy(name):
    def register_sweeper_fn(cls):
        if name in __SWEEP_STRATEGY_DICT__:
            raise ValueError(f'Sweep Strategy {name} already registered!')
        __SWEEP_STRATEGY_DICT__[name] = cls
        return cls

    return register_sweeper_fn

def create_sweep_strategy_from_config(config):
    sweep_strat = config['sweep_configuration']['method']
    if sweep_strat in __SWEEP_STRATEGY_DICT__:
        return __SWEEP_STRATEGY_DICT__[sweep_strat].create_from_config(config)
    else:
        raise ValueError(f'Error! Unrecognized sweep_strategy {sweep_strat}!')


class SweepStrategy:

    def __init__(self):
        pass

    def parse_sweep_parameters(self, param_options):
        """The function gets an dictionary containing all options
        to sweep over for every parameter: param_options[NAME] = [ALL OPTIONS].
        It is called during initialization and gives the strategy the possibility to
        parse parameter into their own format.
        """
        raise NotImplementedError('Error, implement parse_sweep_parameters within your strategy!')


@register_sweep_strategy('grid')
class GridSweepStrategy(SweepStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def parse_sweep_parameters(self, param_options):
        i = 0
        sweep_parameters = {}
        for k, v in param_options.items():
            sweep_parameters[i] = {}
            sweep_parameters[i]['name'] = k
            sweep_parameters[i]['values'] = v
            i += 1
        
        for i in range(len(sweep_parameters)):
            c = 1
            for j in range(i+1, len(sweep_parameters)):
                c *= len(sweep_parameters[j]['values'])
            sweep_parameters[i]['div'] = c
        
        self.sweep_parameters= sweep_parameters

    @staticmethod
    def create_from_config(config):
        return GridSweepStrategy()
    
    def get_config(self, iteration):
        idx = iteration
        sweep_config = {}
        for k, v in self.sweep_parameters.items():

            div = self.sweep_parameters[k]['div']

            i = idx // div
            idx = idx % div

            name = self.sweep_parameters[k]['name']
            value = self.sweep_parameters[k]['values'][i]

            sweep_config[name] = value
        
        return sweep_config


@register_sweep_strategy('random')
class RandomSweepStrategy(GridSweepStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_from_config(config):
        return RandomSweepStrategy()

    def parse_sweep_parameters(self, param_options):
        self.param_options = param_options

    def get_config(self, iteration):
        config = {}
        for name, values in self.param_options.items():
            config[name] = random.choice(values)
        return config