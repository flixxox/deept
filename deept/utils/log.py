from os.path import join

import torch

from deept.utils.debug import my_print
from deept.utils.globals import Settings


def int_to_str(value):
    return f'{value:0>4}'

def float_to_str(value):
    if value > 1e7:
        value = float('inf')
    return f'{value:4.2f}'

def float_to_str_precise(value):
    if value > 1e8:
        value = float('inf')
    return f'{value:8.6f}'

def tensor_to_float(value):
    return value.cpu().detach().numpy()

def value_to_str(v, no_precise=False):
    if isinstance(v, torch.Tensor):
        v = tensor_to_float(v)
    if isinstance(v, int):
        v = int_to_str(v)
    elif isinstance(v, float):
        if v > 1e-2 or v == 0. or no_precise:
            v = float_to_str(v)
        else:
            v = float_to_str_precise(v)
    return v

def write_number_to_file(filename, value):
    if Settings.rank() == 0:
        with open(join(Settings.get_dir('numbers_dir'), filename), 'a') as file:
            file.write(f'{value}\n')

def print_summary(name, number, **kwargs):
    
    def pr_int(name, value):
        length = len(name) + 2 + 4
        if value is not None:
            return f'{name}: {value:0>4}'
        else:
            return ''.center(length, ' ')

    def pr_float_precise(name, value):
        length = len(name) + 2 + 10 + 6
        if value is not None:
            if value > 1e8:
                value = float('inf')
            return f'{name}: {value:8.6f}'.ljust(length, ' ')
        else:
            return ''.center(length, ' ')

    def pr_float(name, value):
        length = len(name) + 2 + 8 + 2
        if value is not None:
            if value > 1e7:
                value = float('inf')
            return f'{name}: {value:4.2f}'.ljust(length, ' ')
        else:
            return ''.center(length, ' ')

    to_print = (
        f'| {name.center(8, " ")} '
        f'| number: {number:0>4} '
    )

    for k, v in kwargs.items():

        if v is None:
            continue

        if isinstance(v, torch.Tensor):
            v = tensor_to_float(v)

        if isinstance(v, int):
            to_print = (
                f'{to_print}'
                f'| {pr_int(k, v)} '
            )
        elif isinstance(v, float):
            if v > 1e-2 or v == 0.:
                to_print = (
                    f'{to_print}'
                    f'| {pr_float(k, v)} '
                )
            else:
                to_print = (
                    f'{to_print}'
                    f'| {pr_float_precise(k, v)} '
                )

    my_print(to_print)

def write_scores_dict_to_files(scores_dict, prefix=''):
    for k, v in scores_dict.items():
        write_number_to_file(prefix + '.' + k, v)


class ScoreSummary:

    def __init__(self, prefix=''):
        self.prefix = prefix
        self.summaries = []

    def push_new_summary(self):
        self.summaries.append({})

    def update_latest_from_score(self, score):
        values = score.get_reduced_accumulator_values()
        self.summaries[-1].update(values)

    def update_latest_from_key_value(self, key, value):
        self.summaries[-1][key] = value

    def log_latest(self, number):
        write_scores_dict_to_files(self.summaries[-1], prefix=self.prefix)
        print_summary(self.prefix, number, **self.summaries[-1])

    def get_latest(self):
        return self.summaries[-1]