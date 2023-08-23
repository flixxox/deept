import time
from os import mkdir, listdir
from os.path import isdir, isfile, join

import torch

from deept.util.debug import my_print
from deept.util.globals import Settings, Context


class CheckpointManager:

    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.best_score = float('inf')
        self.ckpts_since_best = 0
        self.step_count = 1 # The current step. Increased after do_checkpoint_after_step()
        self.epoch_count = 1 # The current epoch. Increased after do_checkpoint_after_epoch()
        self.checkpoint_count = 1 # The current checkpoint number. Increased after save()
        self.timestamp = 0
        self.checkpoint_duration_accum = 0

    @staticmethod
    def create_train_checkpoint_manager_from_config(config):

        checkpoint_manager = CheckpointManager(
            checkpoint_dir = Settings.get_dir('checkpoint_dir'),
            checkpoint_period = config['checkpoint_period'],
            resume_training = config['resume_training', False],
            do_checkpoints = config['checkpoints'],
            checkpoint_unit = config['checkpoint_unit'],
            units_to_train = config['units_to_train'],
            checkpoint_strategy = config['checkpoint_strategy', 'All'],
            do_early_abort = config['early_abort', False],
            checkpoints_till_abort = config['checkpoints_till_abort', 0],
            checkpoint_start_after = config['checkpoint_start_after', 0],
            best_indicator = config['best_checkpoint_indicator'],
            load_weights = config['load_weights', False],
            load_weights_from = config['load_weights_from', ""],
            strict_loading = config['checkpoint_strict_loading', True] 
        )

        return checkpoint_manager

    @staticmethod
    def create_eval_checkpoint_manager_from_config(config):

        checkpoint_manager = CheckpointManager(
            strict_loading = config['checkpoint_strict_loading', True] 
        )

        return checkpoint_manager


    def restore_if_requested(self):
        if self.resume_training:
            self.restore_latest()
        elif self.load_weights:
            self.load_weights_from_checkpoint()

    def restore_latest(self):
        self.restore(self.get_latest_checkpoint_path())
    
    def load_weights_from_checkpoint(self):

        my_print(f'Loading weights from {self.load_weights_from}!')

        Context['model'].init_weights_from_checkpoint(self.load_weights_from)

    def get_latest_checkpoint_path(self):

        maybe_last_checkpoint_path = join(self.checkpoint_dir, f'ckpt-last.pt')

        if isfile(maybe_last_checkpoint_path):
            return maybe_last_checkpoint_path

        file_names = [f for f in listdir(self.checkpoint_dir) if isfile(join(self.checkpoint_dir, f))]
        max_number = -1

        for file_name in file_names:
            
            if file_name.startswith('ckpt-'):
                number = file_name.split('.')[0].split('-')[1]

                if number.isdigit():
                    number = int(number)

                    if number > max_number:
                        max_number = number

        return join(self.checkpoint_dir, f'ckpt-{max_number}.pt')

    def restore(self, path):

        my_print(f'Loading weights from {path}!')

        checkpoint = torch.load(path, map_location=Settings.get_device())

        Context['model'].load_state_dict(checkpoint['model'], 
            strict=self.strict_loading
        )
        self.best_score = checkpoint['best_score']
        self.ckpts_since_best = checkpoint['ckpts_since_best']
        self.step_count = checkpoint['step_count']
        self.epoch_count = checkpoint['epoch_count']
        self.checkpoint_count = checkpoint['checkpoint_count']+1
        self.checkpoint_duration_accum = checkpoint['checkpoint_duration_accum']

        if Settings.is_training():
            Context['optimizer'].load_state_dict(checkpoint['optimizer'])
        if Settings.is_training():
            Context['lr_scheduler'].load_state_dict(checkpoint['lr_scheduler'])


    def save(self, score_summary):

        if Settings.rank() == 0:

            self.save_last()
            
            cur_score = score_summary[self.best_indicator]

            if cur_score < self.best_score:
                self.save_best()
                self.best_score = cur_score
                self.ckpts_since_best = 0
            else:
                self.ckpts_since_best += 1

            if self.checkpoint_strategy == 'All' and self.ready_to_checkpoint():
                self.save_numbered()

        self.checkpoint_count += 1

    def save_numbered(self):
        my_print('Saving checkpoint!')
        self.__save(join(self.checkpoint_dir, f'ckpt-{self.checkpoint_count}.pt'))

    def save_last(self):
        my_print('Saving last checkpoint!')
        self.__save(join(self.checkpoint_dir, f'ckpt-last.pt'))

    def save_best(self):
        my_print('Saving best checkpoint!')
        self.__save(join(self.checkpoint_dir, f'ckpt-best.pt'))

    def __save(self, path):

        from torch.nn.parallel import DistributedDataParallel as DDP

        model = Context['model']

        if isinstance(model, DDP):
            model = model.module

        torch.save({
            'model': model.state_dict(),
            'optimizer': Context['optimizer'].state_dict(),
            'lr_scheduler': Context['lr_scheduler'].state_dict(),
            'best_score': self.best_score,
            'ckpts_since_best': self.ckpts_since_best,
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'checkpoint_count': self.checkpoint_count,
            'checkpoint_duration_accum': self.checkpoint_duration_accum
        }, path)


    def keep_going(self):

        if self.checkpoint_unit == 'Step':
            unit = self.step_count
        elif self.checkpoint_unit == 'Epoch':
            unit = self.epoch_count
        else:
            raise ValueError(f'Unrecognized checkpoint unit: {self.checkpoint_unit}!')

        if unit <= self.units_to_train and not self.early_abort():
            return True
        else:
            return False

    def early_abort(self):
        
        if not self.do_early_abort:
            return False

        if self.ckpts_since_best >= self.checkpoints_till_abort and self.ready_to_checkpoint():
            return True
        else:
            return False

    def ready_to_checkpoint(self):

        if self.checkpoint_unit == 'Step':
            unit = self.step_count
        else:
            unit = self.epoch_count

        if unit-1 <= self.checkpoint_start_after:
            return False
        else:
            return True


    def do_checkpoint_after_step(self):

        if self.do_checkpoints:
            
            if self.checkpoint_unit == 'Step':

                if self.step_count % self.checkpoint_period == 0:

                    self.step_count += 1
                    return True

        self.step_count += 1
        return False

    def do_checkpoint_after_epoch(self):

        if self.do_checkpoints:
            
            if self.checkpoint_unit == 'Epoch':

                if self.epoch_count % self.checkpoint_period == 0:

                    self.epoch_count += 1
                    return True

        self.epoch_count += 1
        return False
    
    def get_checkpoint_number(self):
        return self.checkpoint_count

    def timer_start(self):
        self.timestamp = time.perf_counter() 

    def timer_end(self):
        checkpoint_duration_s = time.perf_counter() - self.timestamp
        self.checkpoint_duration_accum += checkpoint_duration_s
        return checkpoint_duration_s

    def average_last_N_checkpoints(self, N):

        if Settings.rank() == 0:

            assert N > 1
            assert self.checkpoint_count > 1

            my_print(f'Averaging last {N} checkpoints!')
            
            checkpoint_paths = [join(self.checkpoint_dir, f'ckpt-{i}.pt') for i in range(max(self.checkpoint_count-N, 1), self.checkpoint_count)]

            self.average_checkpoints(checkpoint_paths, suffix='avg-last')

    def average_N_after_best_checkpoint(self, N):

        if Settings.rank() == 0:
        
            assert N > 1
            assert self.checkpoint_count > 1
            
            best_checkpoint = self.checkpoint_count - self.ckpts_since_best - 1

            min_ckpt = max(best_checkpoint, 1)
            max_ckpt = min(best_checkpoint+N, self.checkpoint_count)

            checkpoint_paths = [join(self.checkpoint_dir, f'ckpt-{i}.pt') for i in range(min_ckpt, max_ckpt)]
            
            my_print(f'Averaging [{min_ckpt}, {max_ckpt}) checkpoints!')

            self.average_checkpoints(checkpoint_paths, suffix='avg-best')

    def average_checkpoints(self, checkpoint_paths, suffix='avg'):

        state_dict = None

        for path in checkpoint_paths:
            
            my_print(f'Summing {path}')

            checkpoint = torch.load(path, map_location=Settings.get_device())
            
            if state_dict is None:
                state_dict = checkpoint['model']
                for k, v in state_dict.items():
                    state_dict[k] = v.to(torch.float64)
            else:
                for k, v in checkpoint['model'].items():
                    state_dict[k] += v.to(torch.float64)

        for k in state_dict.keys():
            state_dict[k] = (state_dict[k] / len(checkpoint_paths)).to(torch.float32)

        my_print(f'Saving averaged checkpoint!')

        torch.save({
            'model': state_dict,
            'best_score': 0.,
            'ckpts_since_best': 0,
            'step_count': 0,
            'epoch_count': 0,
            'checkpoint_count': 0,
            'checkpoint_duration_accum': 0.
        }, join(self.checkpoint_dir, f'ckpt-{suffix}.pt'))

        my_print(f'Averaged {len(checkpoint_paths)} checkpoints!')
