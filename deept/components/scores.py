
from os import mkdir

import torch
import torch.nn as nn
import torch.distributed as dist

from deept.util.globals import Settings
from deept.util.debug import my_print, write_number_to_file

__SCORES__ = {}


def register_score(name):
    def register_score_fn(cls):
        if name in __SCORES__:
            raise ValueError(f'Score {name} already registered!')
        __SCORES__[name] = cls
        return cls

    return register_score_fn

def create_score_from_config(score_config, config):
    score_type = score_config['score_type']
    input_keys = score_config['input_keys']
    if score_type in __SCORES__:
        from deept.util.debug import my_print
        score = __SCORES__[score_type].create_from_config(config, input_keys)
        check_score(score)
        return score
    else:
        raise ValueError(f'Error! Unrecognized score {score_type}!')

def check_score(score):

    assert hasattr(score, 'input_keys'), """Every score used with DeepT must have the attribute 'input_keys'.
        'input_keys' can be read from the config during score.create_from_config(config). It is expected to be a list
        of data keys which are fed alongside the model output to the __call__ function of your score."""

def get_all_score_keys():
    return list(__SCORES__.keys())


def write_scores_dict_to_files(scores_dict, prefix=''):
    for k, v in scores_dict.items():
        write_number_to_file(prefix + '.' + k, v)


class ScoreAccummulator:

    def __init__(self, name):

        self.L = torch.tensor([0.], requires_grad=False, device=Settings.get_device())
        self.value = torch.tensor([0.], requires_grad=False, device=Settings.get_device())
        self.name = name
        self.last_averaged_value = 0.

    def increase(self, value, L):
        with torch.no_grad():
            self.L += L.detach()
            self.value += value.detach()

    def average(self):

        if isinstance(self.L, torch.Tensor):
            L = self.__maybe_distribute_and_to_float(self.L)
        else:
            L = self.L

        if isinstance(self.value, torch.Tensor):
            value = self.__maybe_distribute_and_to_float(self.value)
        else:
            value = self.value

        assert L > 0., """Something went wrong. Score accumulator has not accumulated any values.
            If you have registered accumulators in your score (something you need to do),
            you need to increase it within the __call__ of your score."""

        self.last_averaged_value = (value / L)

        return self.last_averaged_value

    def reset(self):
        self.L[0] = 0.
        self.value[0] = 0.

    def __maybe_distribute_and_to_float(self, tensor):
        if Settings.get_number_of_workers() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.cpu().detach().numpy())


class Score(nn.Module):

    def __init__(self):
        super().__init__()

        self.sub_scores = [] # Other Score objects
        self.accumulators = []
            
    def register_accumulator(self, name):
        self.accumulators.append(ScoreAccummulator(name))

    def register_subscore(self, score):
        self.sub_scores.append(score)

    def get_average_accumulator_values(self):

        values = {}

        for sub_score in self.sub_scores:
            sub_score_values = sub_score.average()
            values.update(sub_score_values)

        for accumulator in self.accumulators:
            values[accumulator.name] = accumulator.average()

        return values

    def reset_accumulators(self):
        for accumulator in self.accumulators:
            accumulator.reset()


@register_score('CrossEntropy')
class CrossEntropy(Score):

    def __init__(self, input_keys,
        pad_index=None,
        calculate_ppl=True,
        label_smoothing=0.0
    ):
        super().__init__()

        self.input_keys = input_keys

        self.pad_index = pad_index
        self.calculate_ppl = calculate_ppl
        self.label_smoothing = label_smoothing

        if pad_index is None:
            ignore_index = -100

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=0.0
        )

        self.register_accumulator('ce')

    @staticmethod
    def create_from_config(config, input_keys):

        from deept.util.globals import Context

        if Context.has_context('pad_index'):
            pad_index = Context['pad_index']
        else:
            pad_index = None
        
        return CrossEntropy(
            input_keys,
            pad_index = pad_index,
            calculate_ppl = config['ce_calculate_ppl', True],
            label_smoothing = config['ce_label_smoothing', 0.0],
        )

    def __call__(self, output, targets):
        ce = self.loss_fn(output, targets)
        self.accumulators[0].increase(ce, numel)
        return ce, numel

    def get_average_accumulator_values(self):
        """Overwrite the get_average_accumulator_values function to also calculate ppl."""
        scores = super(CrossEntropy, self).get_average_accumulator_values()
        if self.calculate_ppl:
            scores['ppl'] = self.__calculate_ppl(scores['ce'])
        return scores

    def __calculate_ppl(self, ce):
        import math
        try: 
            ppl = math.exp(ce)
        except OverflowError: 
            ppl = float('inf')
        return ppl


@register_score('Accuracy')
class Accuracy(Score):

    def __init__(self, input_keys, pad_index=None):
        super().__init__()
        self.input_keys = input_keys
        self.pad_index = pad_index
        self.register_accumulator('acc')

    @staticmethod
    def create_from_config(config, input_keys):

        from deept.util.globals import Context

        if Context.has_context('pad_index'):
            pad_index = Context['pad_index']
        else:
            pad_index = None

        return Accuracy(
            input_keys,
            pad_index=pad_index
        )

    def __call__(self, output, targets):

        return 1., 10
