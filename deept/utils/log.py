from os.path import join

import torch

from deept.utils.debug import my_print
from deept.utils.globals import Settings
from deept.components.scores import Score


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

def round_if_float(v):
    if isinstance(v, float):
        return round(v, 4)
    return v

def write_to_file(dir, filename, string):
    if Settings.rank() == 0:
        with open(join(Settings.get_dir(dir), filename), 'a') as file:
            file.write(f'{string}\n')

def write_and_print(dir, filename, string):
    my_print(string)
    write_to_file(dir, filename, string)

def write_number_to_file(filename, value):
    if Settings.rank() == 0:
        with open(join(Settings.get_dir('numbers_dir'), filename), 'a') as file:
            file.write(f'{value_to_str(value)}\n')

def write_scores_dict_to_files(scores_dict, prefix=''):
    for k, v in scores_dict.items():
        write_number_to_file(prefix + '.' + k, v)

def print_summary(name, number, **kwargs):
    
    def pr_int(name, value):
        length = len(name) + 2 + 6
        if value is not None:
            return f'{name}: {value:0>4}'
        else:
            return ''.center(length, ' ')

    def pr_float_precise(name, value):
        length = len(name) + 10
        if value is not None:
            if value > 1e8:
                value = float('inf')
            return f'{name}: {value:4.4f}'.ljust(length, ' ')
        else:
            return ''.center(length, ' ')

    def pr_float(name, value):
        length = len(name) + 8
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

def write_dict_to_yaml(file_path, to_log):
    import hiyapyco
    with open(file_path, 'w') as file:
        file.write(hiyapyco.dump(to_log))


class Summary:

    def __init__(self, prefix):
        self.prefix = prefix
        self.__summary = {} 

    def update_from_score(self, score):
        assert isinstance(score, Score)
        update = score.get_reduced_accumulator_values()
        self.__summary.update(update)

    def update_from_key_value(self, key, value):
        self.__summary[key] = value

    def get_value(self, key):
        return self.__summary[key]

    def asdict(self):
        return self.__summary

    def items(self):
        return self.__summary.items()

    def keys(self):
        return self.__summary.keys()

    def values(self):
        return self.__summary.values()

    def log(self, number, write_to_file):
        if write_to_file:
            write_scores_dict_to_files(self.__summary, prefix=self.prefix)
        print_summary(self.prefix, number, **self.__summary)
        if Settings.use_wandb():
            self.wandb_log()

    def wandb_log(self):
        import wandb
        to_log = {}
        for k,v in self.__summary.items():
            if self.prefix not in k:
                to_log[f'{self.prefix}_{k}'] = v
            else:
                to_log[k] = v
        wandb.log(to_log)

    def log_to_yaml(self, output_file_path, best_ckpt_id):
        to_log = {}
        to_log['best_ckpt'] = best_ckpt_id
        for k, v in self.__summary.items():
            to_log[k] = round_if_float(v)
        write_dict_to_yaml(output_file_path, to_log)

class SummaryManager:

    def __init__(self,
        best_indicator=None,
        reduce_fn=None,
        prefix=''
    ):
        self.best_ind = best_indicator
        self.reduce_fn = reduce_fn
        self.prefix = prefix
        self.summaries = []

    def push_new_summary(self):
        self.summaries.append(Summary(self.prefix))

    def update_latest_from_score(self, score):
        self.get_latest().update_from_score(score)

    def update_latest_from_key_value(self, key, value):
        self.get_latest().update_from_key_value(key, value)

    def log_latest(self, write_to_file=True):
        self.get_latest().log(len(self.summaries), write_to_file)

    def get_summary_of_best(self):
        x = [summary.get_value(self.best_ind) for summary in self.summaries]
        best = self.reduce_fn(x)
        best_ckpt_idx = x.index(best)
        return best_ckpt_idx, self.summaries[best_ckpt_idx]

    def get_best_value(self, best_key, reduce_fn):
        return reduce_fn([summary.get_value(best_key) for summary in self.summaries])

    def get_latest(self):
        return self.summaries[-1]

    def get_by_index(self, idx):
        return self.summaries[idx]

    def log_best_to_yaml(self):
        from os.path import join
        best_ckpt_idx, summary_of_best = self.get_summary_of_best()
        output_dir = Settings.get_dir('output_dir')
        output_dir = join(output_dir, f'best_{self.prefix}.yaml')
        summary_of_best.log_to_yaml(output_dir, best_ckpt_idx+1)

    def write_best_so_far_to_latest(self):
        best_score = self.get_best_value(self.best_ind, self.reduce_fn)
        self.summaries[-1].update_from_key_value('best_score', best_score)
