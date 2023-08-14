import math
from contextlib import nullcontext

import torch

from deept.util.timer import ContextTimer
from deept.util.globals import Settings, Context
from deept.model.scores import write_scores_dict_to_files
from deept.util.debug import (
    my_print,
    print_summary,
    print_memory_usage,
    write_number_to_file
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

        if Trainer.do_ddp():
            self.model_input_keys = self.model.module.input_keys
        else:
            self.model_input_keys = self.model.input_keys

    @staticmethod
    def create_trainer_from_config(config, train_dataloader, dev_dataloader, checkpoint_manager):

        trainer = Trainer(
            train_dataloader = train_dataloader,
            dev_dataloader = dev_dataloader,
            checkpoint_manager = checkpoint_manager,
            update_freq = config['update_freq'],
            allow_none_type_gradients = config['allow_none_type_gradients', False],
            deterministic = config['deterministic', False]
        )

        return trainer

    @staticmethod
    def do_ddp():
        return Settings.get_number_of_workers() > 1

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

                if self.checkpoint_manager.do_checkpoint_after_step():
                    self.do_checkpoint()

                    if not self.checkpoint_manager.keep_going():
                        return

                data_loading_timer.start()

            if self.checkpoint_manager.do_checkpoint_after_epoch():
                self.do_checkpoint()
    
    def do_checkpoint(self):

        time_passed_s = self.checkpoint_manager.timer_end()
        checkpoint_number = self.checkpoint_manager.get_checkpoint_number()

        score_summary = self.create_score_summary_dict()
        write_scores_dict_to_files(score_summary, prefix='train')
        score_summary['train_steps'] = self.checkpoint_manager.step_count-1
        print_summary(True, checkpoint_number, **score_summary)

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

        score_summary = self.create_score_summary_dict()
        write_scores_dict_to_files(score_summary, prefix='dev')
        score_summary['eval_steps'] = steps

        print_summary(False, self.checkpoint_manager.get_checkpoint_number(), **score_summary)

        self.model.train()

        return score_summary

    def train_step(self, data):

        L_accum = 0

        with self.model.no_sync() if Trainer.do_ddp() else nullcontext():
            for i in range(len(data)-1):
                L = self.train_ministep(data[i])
                L_accum += L
        
        L_accum += self.train_ministep(data[-1])
        
        with ContextTimer('write_L_to_file'):
            write_number_to_file('L', L_accum)

        with ContextTimer('average_gradients'):
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad /= L_accum
                else:
                    if not self.allow_none_type_gradients:
                        raise RuntimeError(f'Detected NoneType gradient!')

        with ContextTimer('optimizer_step'):
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return L_accum
    
    def train_ministep(self, data):
        
        with ContextTimer('send_to_gpu'):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] =  data[k].to(Settings.get_device())

        with ContextTimer('model'):
            output, _ = self.model(*[data[k] for k in self.model_input_keys])

        with ContextTimer('criterion'):
            criterion, L = self.criterion(output, *[data[k] for k in self.criterion.input_keys])

        with ContextTimer('backpropagation'):
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

        with torch.no_grad():
            output, _ = self.model(*[data[k] for k in self.model_input_keys])
            self.criterion(output, *[data[k] for k in self.criterion.input_keys])
            for score in self.scores:
                score(output, *[data[k] for k in score.input_keys])

    def create_score_summary_dict(self):

        score_summary = {}

        criterion_values = self.criterion.average_and_reset_accumulators()
        score_summary.update(criterion_values)

        for score in self.scores:
            score_summary.update(score.average_and_reset_accumulators())

        return score_summary