import math
from contextlib import nullcontext

import torch
from torch import autocast
from torch.cuda import amp
import torch.distributed as dist

from deept.utils.globals import Settings, Context
from deept.utils.log import SummaryManager, write_number_to_file
from deept.utils.debug import (
    my_print,
    print_memory_usage,
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

        reduce_fn = self.checkpoint_manager.ckpt_strategy_to_reduce_fn()

        self.train_summary_manger = SummaryManager(
            best_indicator=self.best_indicator,
            reduce_fn=reduce_fn,
            prefix='train'
        )
        self.eval_summary_manger = SummaryManager(
            best_indicator=self.best_indicator,
            reduce_fn=reduce_fn,
            prefix='eval'
        )
        if self.print_per_step_summary:
            self.step_summary_manger = SummaryManager(prefix='step')

        if Trainer.is_ddp():
            self.model_input_keys = self.model.module.input_keys
        else:
            self.model_input_keys = self.model.input_keys

        if self.wandb_log_grad_norms:
            import wandb

    @staticmethod
    def create_trainer_from_config(config, train_dataloader, dev_dataloader, checkpoint_manager):
        trainer = Trainer(
            train_dataloader = train_dataloader,
            dev_dataloader = dev_dataloader,
            checkpoint_manager = checkpoint_manager,
            num_workers = Settings.get_number_of_workers(),
            update_freq = config['update_freq'],
            best_indicator = config['best_checkpoint_indicator'],
            allow_none_type_gradients = config['allow_none_type_gradients', False],
            deterministic = config['deterministic', False],
            mixed_precision_training = config['mixed_precision_training', False],
            print_per_step_summary = config['print_per_step_summary', False],
            print_per_step_mem_usage = config['print_per_step_mem_usage', False],
            average_gradients = config['average_gradients', True],
            wandb_log_grad_norms = config['wandb_log_grad_norms', False],
            wandb_log_grad_norms_param_filter = config['wandb_log_grad_norms_param_filter', ''],
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

                if hasattr(self.model, 'train_step_end_callback'):
                    self.model.train_step_end_callback(
                        self.checkpoint_manager.step_count+1,
                    )

                if self.print_per_step_summary:
                    self.print_step_summary()

                if self.print_per_step_mem_usage:
                    print_memory_usage()

                if self.checkpoint_manager.do_checkpoint_after_step():
                    self.do_checkpoint()

                    if not self.checkpoint_manager.keep_going():
                        break

            for scheduler in self.lr_schedulers: scheduler.epoch()

            if hasattr(self.model, 'train_epoch_end_callback'):
                self.model.train_epoch_end_callback(
                    self.checkpoint_manager.epoch_count
                )

            if self.checkpoint_manager.do_checkpoint_after_epoch():
                self.do_checkpoint()

        self.train_summary_manger.log_best_to_yaml()
        self.eval_summary_manger.log_best_to_yaml()

        return {
            'train': self.train_summary_manger,
            'eval': self.eval_summary_manger
        }
    
    def do_checkpoint(self):
        time_passed_s = self.checkpoint_manager.timer_end()

        self.create_fill_checkpoint_summary(
            self.train_summary_manger,
            self.checkpoint_manager.step_count-1,
            'train'
        )
        
        self.train_summary_manger.log_latest(
            self.checkpoint_manager.get_checkpoint_number()
        )

        self.eval()

        print_memory_usage()
        my_print(f'Training checkpoint took: {time_passed_s:4.2f}s, {time_passed_s / 60:4.2f}min')

        self.checkpoint_manager.save(self.eval_summary_manger.get_latest())

        self.checkpoint_manager.timer_start()

    def eval(self):
        self.model.eval()

        if hasattr(self.model, 'test_start_callback'):
            self.model.test_start_callback()

        steps = 0
        for data in self.dev_dataloader:
            self.eval_step(data)
            steps += 1

        if hasattr(self.model, 'test_end_callback'):
            self.model.test_end_callback()

        self.create_fill_checkpoint_summary(
            self.eval_summary_manger,
            steps,
            'eval'
        )

        self.eval_summary_manger.log_latest(
            self.checkpoint_manager.get_checkpoint_number()
        )

        self.model.train()

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
            if p.requires_grad:
                if p.grad is not None:
                    if self.average_gradients:
                        p.grad *= (self.num_workers/L_accum)
                else:
                    if not self.allow_none_type_gradients:
                        p_names = search_name_of_parameter(self.model, p)
                        raise RuntimeError(f'Detected NoneType gradient! Name {p_names}, Shape {p.shape}, {p}')

        if self.wandb_log_grad_norms:
            param_filter = self.wandb_log_grad_norms_param_filter
            norm_dict = {}
            for name, p in self.model.named_parameters():
                if p.requires_grad and (param_filter in name):
                    norm_dict[name] = torch.linalg.norm(p.grad)
            if len(norm_dict) > 0:
                wandb.log(norm_dict)
            else:
                my_print('Warning! No matching parameter that can be logged to wandb!')

        if hasattr(self.model, 'callback_optimizer_step_begin'):
            self.model.callback_optimizer_step_begin()

        for opt in self.optimizers: opt.step()
        for scheduler in self.lr_schedulers: scheduler.step()

        if hasattr(self.model, 'callback_optimizer_step_end'):
            self.model.callback_optimizer_step_end()

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
        self.create_fill_checkpoint_summary(
            self.step_summary_manger,
            self.checkpoint_manager.step_count-1,
            'step',
            reset=False
        )
        self.step_summary_manger.log_latest(
            self.checkpoint_manager.get_checkpoint_number(),
            write_to_file=False
        )

    def create_fill_checkpoint_summary(self, summary_manager, steps, prefix, reset=True):
        summary_manager.push_new_summary()

        for criterion in self.criterions:
            summary_manager.update_latest_from_score(criterion)

        for score in self.scores:
            summary_manager.update_latest_from_score(score)

        summary_manager.update_latest_from_key_value(
            f'{prefix}_steps',
            steps
        )

        summary_manager.write_best_so_far_to_latest()

        if reset:
            self.reset_accumulators()

    def reset_accumulators(self):
        for criterion in self.criterions:
            criterion.reset_accumulators()
        for score in self.scores:
            score.reset_accumulators()
