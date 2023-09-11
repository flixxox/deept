
from os import mkdir

import torch
import torch.nn as nn
import torch.distributed as dist

from deept.util.globals import Settings
from deept.util.debug import write_number_to_file


__SCORES__ = {}


def register_score(name):
    def register_score_fn(cls):
        if name in __SCORES__:
            raise ValueError(f'Score {name} already registered!')
        __SCORES__[name] = cls
        return cls

    return register_score_fn

def create_score_from_config(key, config):
    if key in __SCORES__:
        from deept.util.debug import my_print
        score = __SCORES__[key].create_from_config(config)
        check_score(score)
        return score
    else:
        raise ValueError(f'Error! Unrecognized score {key}!')

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

        self.L = 0.
        self.value = 0.
        self.name = name
        self.last_averaged_value = 0.

    def increase(self, value, L):
        self.L += L
        self.value += value

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
        self.L = 0.
        self.value = 0.

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
        calculate_ppl=True
    ):
        super().__init__()

        self.input_keys = input_keys

        self.pad_index = pad_index
        self.calculate_ppl = calculate_ppl

        self.register_accumulator('ce')

    @staticmethod
    def create_from_config(config):

        from deept.util.globals import Context

        if Context.has_context('pad_index'):
            pad_index = Context['pad_index']
        else:
            my_print(f"""Info: "pad_index" has not been set in Context.
                CrossEntropy does not exclude padding symbols.""")
            pad_index = None
        
        return CrossEntropy(
            config['cross_entropy_input', ['out']],
            pad_index = pad_index,
            calculate_ppl = config['ce_calculate_ppl', True]
        )

    def __call__(self, output, out):

        out = out.reshape(-1, 1)
        output = output.reshape(-1, output.shape[-1])

        out_mask = (out != self.pad_index)

        ce = -1 * output.gather(dim=-1, index=out)
        ce = ce * out_mask

        num_words = out_mask.sum()
        ce = ce.sum()

        self.accumulators[0].increase(ce, num_words)

        return ce, num_words

    def get_average_accumulator_values(self):
        """We overwrite the get_average_accumulator_values function to also calculate ppl."""
        scores = super(CrossEntropy, self).get_average_accumulator_values()
        if self.calculate_ppl:
            scores['ppl_smooth'] = self.__calculate_ppl(scores['ce'])
        return scores

    def __calculate_ppl(self, ce):
        import math
        try: 
            ppl = math.exp(ce)
        except OverflowError: 
            ppl = float('inf')
        return ppl


@register_score('LabelSmoothingCrossEntropy')
class LabelSmoothingCrossEntropyLoss(Score):

    def __init__(self, input_keys, m,
        pad_index=None, 
        calculate_ppl=True
    ):
        super().__init__()

        self.m = m
        self.input_keys = input_keys

        self.pad_index = pad_index
        self.calculate_ppl = calculate_ppl

        self.register_accumulator('ce_smooth')

    @staticmethod
    def create_from_config(config):

        from deept.util.globals import Context

        if Context.has_context('pad_index'):
            pad_index = Context['pad_index']
        else:
            my_print(f"""Info: "pad_index" has not been set in Context.
                LabelSmoothingCrossEntropyLoss does not exclude padding symbols.""")
            pad_index = None

        return LabelSmoothingCrossEntropyLoss(
            config['ls_cross_entropy_input', ['out']],
            config['label_smoothing', 0.1],
            pad_index=pad_index,
            calculate_ppl = config['ce_smooth_calculate_ppl', True]
        )

    def __call__(self, output, out):

        tgtV = output.shape[-1]

        out = out.reshape(-1, 1)
        output = output.reshape(-1, tgtV)

        out_mask = (out != self.pad_index)
        
        m = self.m
        w = m / (tgtV - 1)

        nll_loss = -1 * output.gather(dim=-1, index=out)
        smo_loss = -1 * output.sum(-1, keepdim=True)
        
        ce_smooth = (1 - m - w) * nll_loss + w * smo_loss
        ce_smooth = ce_smooth * out_mask

        L = out_mask.sum()
        ce_smooth = ce_smooth.sum()

        self.accumulators[0].increase(ce_smooth, L)

        return ce_smooth, L

    def get_average_accumulator_values(self):
        """We overwrite the get_average_accumulator_values function to also calculate ppl."""
        scores = super(LabelSmoothingCrossEntropyLoss, self).get_average_accumulator_values()
        if self.calculate_ppl:
            scores['ppl_smooth'] = self.__calculate_ppl(scores['ce_smooth'])
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

        self.register_accumulator('accuracy')

    @staticmethod
    def create_from_config(config):

        from deept.util.globals import Context

        if Context.has_context('pad_index'):
            pad_index = Context['pad_index']
        else:
            my_print(f"""Info: "pad_index" has not been set in Context.
                LabelSmoothingCrossEntropyLoss does not exclude padding symbols.""")
            pad_index = None

        return Accuracy(
            config['accuracy_input', ['out']],
            pad_index=pad_index
        )

    def __call__(self, output, out):

        return 1., 10
