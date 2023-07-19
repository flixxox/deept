
import argparse
from os.path import join

import torch

from deept.util.config import Config
from deept.model.scores import Score
from deept.util.trainer import Trainer
from deept.util.globals import Settings, Context
from deept.util.data import create_datapipe_from_config
from deept.util.debug import my_print, print_memory_usage
from deept.util.checkpoint_manager import CheckpointManager
from deept.model.model import create_model_from_config
from deept.model.optimizer import create_optimizer_from_config
from deept.model.lr_scheduler import create_lr_scheduler_from_config
from deept.util.setup import (
    setup,
    import_user_code,
    check_and_correct_requested_number_of_gpus
)


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the config.yaml which contains all user defined parameters.')
    parser.add_argument('--user-code', type=str, required=True,
        help="""A path to the directory containing the user code.
            The directory must be named 'deept_user'.
            All <NAME>.py files in this directory will be imported as deept_user.<NAME>.
            At the moment, no nesting of folders is supported."""
        )
    parser.add_argument('--output-folder', type=str, required=True, 
        help='The folder in which to write the training output (ckpts, learning-rates, perplexities etc.)')
    parser.add_argument('--resume-training', type=int, required=False, default=False, 
        help='If you want to resume a training, set this flat to 1 and specify the directory with "resume-training-from".')
    parser.add_argument('--resume-training-from', type=str, required=False, default='', 
        help='If you want to resume a training, specify the output directory here. We expect it to have the same layout as a newly created one.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None,
        help='This is usually specified in the config but can also be overwritten from the cli.')

    args = parser.parse_args()

    args.resume_training = bool(args.resume_training)

    return vars(args)

def start(config):

    check_and_correct_requested_number_of_gpus(config)

    world_size = max(1, config['number_of_gpus'])

    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(
            train,
            args=(config, world_size, ),
            nprocs=world_size
        )
    else:
        train(0, config, 1)

def train(rank, config, world_size):

    import_user_code(config['user_code'])

    if config['resume_training']:
        config['output_folder'] = join(config['resume_training_from'])

    setup(config, rank, world_size, train=True)

    if Settings.get_number_of_workers() > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(Settings.get_global_seed())
    
    train_datapipe = create_datapipe_from_config(config, train=True)
    dev_datapipe = create_datapipe_from_config(config, train=False)

    model = create_model_from_config(config, vocab_src, vocab_tgt)
    model.init_weights()
    model = model.to(Settings.get_device())
    model = DDP(model, device_ids=[Settings.rank()])
    Context.add_context('model', model)

    criterion = Score.create_score_from_config(config).to(Settings.get_device())
    criterion = criterion.to(Settings.get_device())
    Context.add_context('criterion', criterion)

    optimizer = create_optimizer_from_config(config, model.parameters())
    Context.add_context('optimizer', optimizer)

    lr_scheduler = create_lr_scheduler_from_config(config, optimizer)
    Context.add_context('lr_scheduler', lr_scheduler)

    my_print(f'Trainable variables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    checkpoint_manager = CheckpointManager.create_train_checkpoint_manager_from_config(config)
    checkpoint_manager.restore_if_requested()

    trainer = Trainer.create_trainer_from_config(config, train_datapipe, dev_datapipe, checkpoint_manager)

    print_memory_usage()
    my_print(f'Start training at checkpoint {checkpoint_manager.get_checkpoint_number()}!')

    trainer.train()

    if config['average_last_checkpoints', False]:
        checkpoint_manager.average_last_N_checkpoints(config['checkpoints_to_average'])

    if config['average_last_after_best_checkpoints', False]:
        checkpoint_manager.average_N_after_best_checkpoint(config['checkpoints_to_average'])

    average_time_per_checkpoint_s = checkpoint_manager.checkpoint_duration_accum / checkpoint_manager.checkpoint_count
    my_print(f'Average time per checkpoint: {average_time_per_checkpoint_s:4.2f}s {average_time_per_checkpoint_s/60:4.2f}min')

    my_print('Done!')


if __name__ == '__main__':

    my_print(''.center(40, '-'))
    my_print(' Hi! '.center(40, '-'))
    my_print(' Script: train.py '.center(40, '-'))
    my_print(''.center(40, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)

    config.print_config()

    start(config)