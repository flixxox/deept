import math

import torch

from deept.model.scores import Score
from deept.util.globals import Settings, Context
from deept.util.timer import Timer, ContextTimer
from deept.util.debug import my_print, print_summary, print_memory_usage


class Trainer:

    def __init__(self, **kwargs):
        super(Trainer, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.model = Context['model']
        self.criterion = Context['criterion']
        self.optimizer = Context['optimizer']
        self.lr_scheduler = Context['lr_scheduler']

    @staticmethod
    def create_trainer_from_config(config, train_datapipe, dev_datapipe, checkpoint_manager):

        trainer = Trainer(
            train_datapipe = train_datapipe,
            dev_datapipe = dev_datapipe,
            checkpoint_manager = checkpoint_manager,
            update_freq = config['update_freq'],
            allow_none_type_gradients = config['allow_none_type_gradients', False],
            deterministic = config['deterministic', False]
        )

        return trainer

    def train(self):

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        self.checkpoint_manager.timer_start()

        while self.checkpoint_manager.keep_going():

            for _, data in self.train_datapipe:

                assert len(data) == self.update_freq

                L = self.train_step(data)

                if self.checkpoint_manager.do_checkpoint_after_step():
                    self.do_checkpoint()

                    if not self.checkpoint_manager.keep_going():
                        return

            if self.checkpoint_manager.do_checkpoint_after_epoch():
                self.do_checkpoint()
    
    def do_checkpoint(self):
        with torch.no_grad():
            time_passed_s = self.checkpoint_manager.timer_end()
            checkpoint_number = self.checkpoint_manager.get_checkpoint_number()

            train_ce, train_ce_smooth = self.criterion.average_and_reset()
            train_ppl, train_ppl_smooth = self.__calculate_ppl(train_ce, train_ce_smooth)

            to_print = {
                'ce': train_ce,
                'ce_smooth': train_ce_smooth,
                'ppl': train_ppl,
                'ppl_smooth': train_ppl_smooth,
                'train_steps': self.checkpoint_manager.step_count-1
            }

            print_summary(True, checkpoint_number, **to_print)

            dev_ppl = self.eval(checkpoint_number)

            print_memory_usage()
            my_print(f'Training checkpoint took: {time_passed_s:4.2f}s, {time_passed_s / 60:4.2f}min')

            self.checkpoint_manager.save(dev_ppl)

            Score.write_score_to_file(self.numbers_dir, 'train_ppl', train_ppl)
            Score.write_score_to_file(self.numbers_dir, 'dev_ppl',   dev_ppl)

            if Settings.do_timing():
                model_time = Timer.print_timing_summary(self.model)
                ContextTimer.print_summary(model_time)

            self.checkpoint_manager.timer_start()

    def eval(self, checkpoint_number):

        self.model.eval()

        for _, data in self.dev_datapipe:

            ce, ce_smooth, _ = self.eval_step(data)

        ce, ce_smooth = self.criterion.average_and_reset()
        ppl, ppl_smooth = self.__calculate_ppl(ce, ce_smooth)

        to_print = {
            'ce':               ce,
            'ce_smooth':        ce_smooth,
            'ppl':              ppl,
            'ppl_smooth':       ppl_smooth,
            'eval_steps':       step
        }

        print_summary(False, checkpoint_number, **to_print)

        self.model.train()

        return ppl

    def __calculate_ppl(self, ce, ce_smooth):

        try: 
            ppl = math.exp(ce) 
            ppl_smooth = math.exp(ce_smooth)

        except OverflowError: 
            ppl = float('inf')
            ppl_smooth = float('inf')

        return ppl, ppl_smooth

    def train_step(self, data):

        L_accum = 0

        with self.model.no_sync():
            for i in range(len(data)-1):
                L = self.train_ministep(data[i])
                L_accum += L
        
        L = self.train_ministep(data[-1])
        L_accum += L

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

        for i in range(len(data)):
            data[i] =  data[i].to(Settings.get_device())

        with ContextTimer('model_mask_creation'):
            masks, out_mask = self.model.module.create_masks(*data)

        output, _ = self.model(*data, **masks)

        with ContextTimer('criterion'):
            _, ce_smooth, L_ce = self.criterion(output, *data, out_mask=out_mask)

        with ContextTimer('backpropagation'):
            ce_smooth.backward()

        if self.deterministic:
            Settings.increase_global_seed()
            torch.manual_seed(Settings.get_global_seed())

        return L_ce

    def eval_step(self, data):
        
        src = src.to(Settings.get_device())
        tgt = tgt.to(Settings.get_device())
        out = out.to(Settings.get_device())

        with torch.no_grad():
            masks, out_mask     = self.model.create_masks(src, out, self.pad_index)
            output, _           = self.model(src, tgt, **masks)
            ce, ce_smooth, L_ce = self.criterion(output, out, out_mask=out_mask)

        return ce, ce_smooth, L_ce