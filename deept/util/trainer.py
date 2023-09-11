import math
from contextlib import nullcontext

import torch
from torch import autocast
from torch.cuda import amp

from deept.util.timer import ContextTimer
from deept.util.globals import Settings, Context
from deept.model.scores import write_scores_dict_to_files
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
        self.criterion = Context['criterion']
        self.optimizer = Context['optimizer']
        self.lr_scheduler = Context['lr_scheduler']
        self.scores = Context['scores']

        if Trainer.is_ddp():
            self.model_input_keys = self.model.module.input_keys
        else:
            self.model_input_keys = self.model.input_keys

        if self.is_mpt():
            my_print('Using mixed_precision training!')
            self.scaler = amp.GradScaler()

    @staticmethod
    def create_trainer_from_config(config, train_dataloader, dev_dataloader, checkpoint_manager):

        trainer = Trainer(
            train_dataloader = train_dataloader,
            dev_dataloader = dev_dataloader,
            checkpoint_manager = checkpoint_manager,
            update_freq = config['update_freq'],
            allow_none_type_gradients = config['allow_none_type_gradients', False],
            deterministic = config['deterministic', False],
            mixed_precision_training = config['mixed_precision_training', False],
            print_per_step_summary = config['print_per_step_summary', False],
            print_per_step_mem_usage = config['print_per_step_mem_usage', False]
        )

        return trainer

    @staticmethod
    def is_ddp():
        return Settings.get_number_of_workers() > 1

    def is_mpt(self):
        return Settings.is_gpu() and self.mixed_precision_training

    def train(self):

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        self.train_dataloader.seed(self.checkpoint_manager.epoch_count)

        self.checkpoint_manager.timer_start()

        data_loading_timer = ContextTimer('data_loading')
        data_loading_timer.start()

        while self.checkpoint_manager.keep_going():

            for data in self.train_dataloader:

                data_loading_timer.end()

                assert len(data) == self.update_freq

                L = self.train_step(data)

                if self.print_per_step_summary:
                    self.print_step_summary()

                if self.print_per_step_mem_usage:
                    print_memory_usage()

                if self.checkpoint_manager.do_checkpoint_after_step():
                    self.do_checkpoint()

                    if not self.checkpoint_manager.keep_going():
                        return

                data_loading_timer.start()

            if self.checkpoint_manager.do_checkpoint_after_epoch():
                self.do_checkpoint()
    
    def do_checkpoint(self):

        time_passed_s = self.checkpoint_manager.timer_end()

        self.do_train_epoch_summary()

        eval_score_summary = self.eval()

        print_memory_usage()
        my_print(f'Training checkpoint took: {time_passed_s:4.2f}s, {time_passed_s / 60:4.2f}min')

        self.checkpoint_manager.save(eval_score_summary)

        if Settings.do_timing():
            ContextTimer.print_summary()

        self.checkpoint_manager.timer_start()

    def eval(self):

        self.model.eval()

        self.dev_dataloader.seed(self.checkpoint_manager.epoch_count)

        steps = 0
        for data in self.dev_dataloader:
            self.eval_step(data)
            steps += 1

        self.do_eval_epoch_summary(steps)

        self.model.train()

        return score_summary

    def train_step(self, data):
        # TODO: A couple of problems with multi-gpu training
        # 1. DDP averages the gradients after synchronizaiton
        # so I need to multiply every gradient by the number of workers
        # to get the same gradient as with one GPU
        # 2. I do not distribute L_accum. Every worker needs to have
        # the same L_accum!

        L_accum = 0

        with self.model.no_sync() if Trainer.is_ddp() else nullcontext():
            for i in range(len(data)-1):
                L = self.train_ministep(data[i])
                L_accum += L

        L_accum += self.train_ministep(data[-1])
        
        with ContextTimer('write_L_to_file'):
            write_number_to_file('L', L_accum)

        if self.is_mpt():
            self.scaler.unscale_(self.optimizer)

        with ContextTimer('average_gradients'):
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad /= L_accum
                else:
                    if not self.allow_none_type_gradients:
                        p_names = search_name_of_parameter(self.model, p)
                        raise RuntimeError(f'Detected NoneType gradient! Name {p_names}, Shape {p.shape}, {p}')

        with ContextTimer('optimizer_step'):
            if self.is_mpt():
                self.scaler.step(self.optimizer)
                self.scaler.update() 
            else:
                self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return L_accum
    
    def train_ministep(self, data):
        
        with ContextTimer('send_to_gpu'):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] =  data[k].to(Settings.get_device())

        with autocast(
            device_type='cuda', dtype=torch.float16
        ) if self.is_mpt() else nullcontext():

            with ContextTimer('model'):
                output, _ = self.model(*[data[k] for k in self.model_input_keys])

            with ContextTimer('criterion'):
                criterion, L = self.criterion(output, *[data[k] for k in self.criterion.input_keys])

        with ContextTimer('backpropagation'):
            if self.is_mpt():
                self.scaler.scale(criterion).backward()
            else:
                criterion.backward()

        with ContextTimer('scores'):
            with torch.no_grad():
                for score in self.scores:
                    score(output, *[data[k] for k in score.input_keys])

        with ContextTimer('setting_seeds'):
            if self.deterministic:
                Settings.increase_global_seed()
                torch.manual_seed(Settings.get_global_seed())

        return L

    def eval_step(self, data):
        
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] =  data[k].to(Settings.get_device())

        with torch.no_grad(), autocast(
            device_type='cuda', dtype=torch.float16
        ) if self.is_mpt() else nullcontext():
            output, _ = self.model(*[data[k] for k in self.model_input_keys])
            self.criterion(output, *[data[k] for k in self.criterion.input_keys])
            for score in self.scores:
                score(output, *[data[k] for k in score.input_keys])

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

    def create_score_summary_dict(self):

        score_summary = {}
        criterion_values = self.criterion.get_average_accumulator_values()
        score_summary.update(criterion_values)

        for score in self.scores:
            score_summary.update(score.get_average_accumulator_values())

        return score_summary

    def reset_accumulators(self):
        self.criterion.reset_accumulators()
        for score in self.scores:
            score.reset_accumulators()
