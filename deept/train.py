
import argparse
from os.path import join

import torch

from deept.util.trainer import Trainer
from deept.util.globals import Settings, Context
from deept.util.datapipe import create_dp_from_config
from deept.model.model import create_model_from_config
from deept.model.scores import create_score_from_config
from deept.util.debug import my_print, print_memory_usage
from deept.util.checkpoint_manager import CheckpointManager
from deept.model.optimizer import create_optimizer_from_config
from deept.util.dataloader import create_dataloader_from_config
from deept.model.lr_scheduler import create_lr_scheduler_from_config
from deept.util.config import (
    Config,
    DeepTConfigDescription
)
from deept.util.setup import (
    setup,
    import_user_code,
    check_and_correct_requested_number_of_gpus
)


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the config.yaml which contains all user defined parameters.')
    parser.add_argument('--user-code', type=str, nargs='+', required=False, default=None,
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

    setup(config, rank, world_size, train=True)

    config['update_freq'] = config['update_freq'] // Settings.get_number_of_workers()
    my_print(f'Scaled down update_freq to {config["update_freq"]}!')

    if config['resume_training']:
        config['output_folder'] = join(config['resume_training_from'])

    torch.manual_seed(Settings.get_global_seed())
    
    train_datapipe = create_dp_from_config(config, 
        config['data_train_root'],
        config['data_train_mask'],
        name='train',
        chunk=True
    )

    dev_datapipe = create_dp_from_config(config,
        config['data_dev_root'],
        config['data_dev_mask'],
        name='dev',
        chunk=False
    )

    train_dataloader = create_dataloader_from_config(config, train_datapipe, shuffle=True)
    dev_dataloader = create_dataloader_from_config(config, dev_datapipe, shuffle=False)

    model = create_model_from_config(config)
    model.init_weights()
    model = model.to(Settings.get_device())
    if config['number_of_gpus'] > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[Settings.get_device()])
    Context.add_context('model', model)

    criterion = create_score_from_config(config['criterion'], config)
    criterion = criterion.to(Settings.get_device())
    Context.add_context('criterion', criterion)

    scores = []
    if config['scores', None] is not None:
        for key in config['scores']:
            score = create_score_from_config(key, config)
            score = score.to(Settings.get_device())
            scores.append(score)
    Context.add_context('scores', scores)

    optimizer = create_optimizer_from_config(config, model.parameters())
    Context.add_context('optimizer', optimizer)

    lr_scheduler = create_lr_scheduler_from_config(config, optimizer)
    Context.add_context('lr_scheduler', lr_scheduler)

    my_print(f'Trainable variables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    checkpoint_manager = CheckpointManager.create_train_checkpoint_manager_from_config(config)
    checkpoint_manager.restore_if_requested()

    trainer = Trainer.create_trainer_from_config(config,
        train_dataloader,
        dev_dataloader,
        checkpoint_manager
    )

    print_memory_usage()
    my_print(f'Start training at checkpoint {checkpoint_manager.get_checkpoint_number()}!')

    trainer.train()

    train_dataloader.shutdown()
    dev_dataloader.shutdown()

    if config['average_last_checkpoints', False]:
        checkpoint_manager.average_last_N_checkpoints(config['checkpoints_to_average'])

    if config['average_last_after_best_checkpoints', False]:
        checkpoint_manager.average_N_after_best_checkpoint(config['checkpoints_to_average'])

    average_time_per_checkpoint_s = checkpoint_manager.checkpoint_duration_accum / checkpoint_manager.checkpoint_count
    my_print(f'Average time per checkpoint: {average_time_per_checkpoint_s:4.2f}s {average_time_per_checkpoint_s/60:4.2f}min')

    my_print('Done!')


if __name__ == '__main__':

    my_print(''.center(60, '-'))
    my_print(' Hi! '.center(60, '-'))
    my_print(' Script: train.py '.center(60, '-'))
    my_print(''.center(60, '-'))

    DeepTConfigDescription.create_deept_config_description()

    args = parse_cli_arguments()

    config = Config.parse_config(args)

    config.print_config()

    start(config)