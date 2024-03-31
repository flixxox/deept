import math
from contextlib import nullcontext

import torch
from torch import autocast
from torch.cuda import amp
import torch.distributed as dist

from deept.util.globals import Settings, Context
from deept.components.scores import write_scores_dict_to_files
from deept.util.debug import (
    my_print,
    print_summary,
    print_memory_usage,
    write_number_to_file,
    search_name_of_parameter
)


class Trainer:

    def __init__(self, **kwargs):
        super(Trainer, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.model = Context['model']
        self.criterions = Context['criterions']
        self.optimizers = Context['optimizers']
        self.lr_schedulers = Context['lr_schedulers']
        self.scores = Context['scores']

        if Trainer.is_ddp():
            self.model_input_keys = self.model.module.input_keys
        else:
            self.model_input_keys = self.model.input_keys

    @staticmethod
    def create_trainer_from_config(config, train_dataloader, dev_dataloader, checkpoint_manager):

        trainer = Trainer(
            train_dataloader = train_dataloader,
            dev_dataloader = dev_dataloader,
            checkpoint_manager = checkpoint_manager,
            num_workers = Settings.get_number_of_workers(),
            update_freq = config['update_freq'],
            allow_none_type_gradients = config['allow_none_type_gradients', False],
            deterministic = config['deterministic', False],
            mixed_precision_training = config['mixed_precision_training', False],
            print_per_step_summary = config['print_per_step_summary', False],
            print_per_step_mem_usage = config['print_per_step_mem_usage', False],
            average_gradients = config['average_gradients', True]
        )

        return trainer

    @staticmethod
    def is_ddp():
        return Settings.get_number_of_workers() > 1

    def train(self):

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        self.checkpoint_manager.timer_start()

        while self.checkpoint_manager.keep_going():

            for data in self.train_dataloader:

                assert len(data) == self.update_freq

                self.train_step(data)

                if self.print_per_step_summary:
                    self.print_step_summary()

                if self.print_per_step_mem_usage:
                    print_memory_usage()

                if self.checkpoint_manager.do_checkpoint_after_step():
                    self.do_checkpoint()

                    if not self.checkpoint_manager.keep_going():
                        return

            for scheduler in self.lr_schedulers: scheduler.epoch()

            if self.checkpoint_manager.do_checkpoint_after_epoch():
                self.do_checkpoint()
    
    def do_checkpoint(self):

        time_passed_s = self.checkpoint_manager.timer_end()

        self.do_train_epoch_summary()

        eval_score_summary = self.eval()

        print_memory_usage()
        my_print(f'Training checkpoint took: {time_passed_s:4.2f}s, {time_passed_s / 60:4.2f}min')

        self.checkpoint_manager.save(eval_score_summary)

        self.checkpoint_manager.timer_start()

    def eval(self):

        self.model.eval()

        steps = 0
        for data in self.dev_dataloader:
            self.eval_step(data)
            steps += 1

        score_summary = self.do_eval_epoch_summary(steps)

        self.model.train()

        return score_summary

    def train_step(self, data):

        for opt in self.optimizers: opt.zero_grad(set_to_none=True)

        L_accum = torch.tensor([0.], requires_grad=False, device=Settings.get_device())

        with self.model.no_sync() if Trainer.is_ddp() else nullcontext():
            for i in range(len(data)-1):
                L = self.train_ministep(data[i])
                L_accum += L

        L_accum += self.train_ministep(data[-1])

        if Trainer.is_ddp():
            dist.all_reduce(L_accum, op=dist.ReduceOp.SUM)
        
        write_number_to_file('L', int(L_accum.cpu().numpy()))

        for p in self.model.parameters():
            if p.grad is not None:
                if self.average_gradients:
                    p.grad *= (self.num_workers/L_accum)
            else:
                if not self.allow_none_type_gradients:
                    p_names = search_name_of_parameter(self.model, p)
                    raise RuntimeError(f'Detected NoneType gradient! Name {p_names}, Shape {p.shape}, {p}')

        for opt in self.optimizers: opt.step()
        for scheduler in self.lr_schedulers: scheduler.step()
    
    def train_ministep(self, data):
        
        for key, tensor in data['tensors'].items():
            data['tensors'][key] = tensor.to(Settings.get_device())

        output, add_tensors = self.model(*[data['tensors'][k] for k in self.model_input_keys])

        data['tensors'].update(add_tensors)

        loss, L = self.criterions[0](
            output, 
            *([data['tensors'][k] for k in self.criterions[0].input_keys])
        )

        for criterion in self.criterions[1:]:
            loss_cur, _ = criterion(output, *[data['tensors'][k] for k in criterion.input_keys])
            loss += loss_cur

        loss.backward()

        with torch.no_grad():
            for score in self.scores:
                score(output, *[data['tensors'][k] for k in score.input_keys])

        if isinstance(L, torch.Tensor):
            L = L.detach()

        return L

    def eval_step(self, data):
        
        for key, tensor in data['tensors'].items():
            data['tensors'][key] = tensor.to(Settings.get_device())

        with torch.no_grad():
            output, add_tensors = self.model(*[data['tensors'].get(k) for k in self.model_input_keys])
            data['tensors'].update(add_tensors)
            for criterion in self.criterions:
                criterion(output, *[data['tensors'].get(k) for k in criterion.input_keys])
            for score in self.scores:
                score(output, *[data['tensors'].get(k) for k in score.input_keys])

    def print_step_summary(self):
        score_summary = self.create_score_summary_dict()
        score_summary['train_steps'] = self.checkpoint_manager.step_count-1
        checkpoint_number = self.checkpoint_manager.get_checkpoint_number()
        print_summary(True, checkpoint_number, **score_summary)

    def do_train_epoch_summary(self):
        checkpoint_number = self.checkpoint_manager.get_checkpoint_number()
        score_summary = self.create_score_summary_dict()
        write_scores_dict_to_files(score_summary, prefix='train')
        score_summary['train_steps'] = self.checkpoint_manager.step_count-1
        print_summary(True, checkpoint_number, **score_summary)
        self.reset_accumulators()

    def do_eval_epoch_summary(self, steps):
        checkpoint_number = self.checkpoint_manager.get_checkpoint_number()
        score_summary = self.create_score_summary_dict()
        write_scores_dict_to_files(score_summary, prefix='dev')
        score_summary['eval_steps'] = steps
        print_summary(False, checkpoint_number, **score_summary)
        self.reset_accumulators()
        return score_summary

    def create_score_summary_dict(self):

        score_summary = {}
        for criterion in self.criterions:
            criterion_values = criterion.get_reduced_accumulator_values()
            score_summary.update(criterion_values)

        for score in self.scores:
            score_summary.update(score.get_reduced_accumulator_values())

        return score_summary

    def reset_accumulators(self):
        for criterion in self.criterions:
            criterion.reset_accumulators()
        for score in self.scores:
            score.reset_accumulators()
