
import argparse
from os.path import join

import torch

from deept.utils.trainer import Trainer
from deept.utils.globals import Settings, Context
from deept.utils.debug import my_print, print_memory_usage
from deept.sweep import Sweeper, create_sweep_strategy_from_config
from deept.utils.config import (
    Config,
    DeepTConfigDescription
)
from deept.utils.setup import (
    setup,
    setup_wandb,
    setup_seeds,
    setup_directories,
    check_and_correct_requested_number_of_gpus
)


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the config.yaml which contains all user defined parameters.')
    parser.add_argument('--user-code', type=str, nargs='+', required=False, default=None,
        help="""One or multiple paths to directories containing user code.""")
    parser.add_argument('--output-folder', type=str, required=True, 
        help='The folder in which to write the training output (ckpts, learning-rates, perplexities etc.)')
    parser.add_argument('--resume-training', type=int, required=False, default=False, 
        help='If you want to resume a training, set this flat to 1 and specify the directory with "resume-training-from".')
    parser.add_argument('--resume-training-from', type=str, required=False, default='', 
        help='If you want to resume a training, specify the output directory here. We expect it to have the same layout as a newly created one.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None,
        help='This is usually specified in the config but can also be overwritten from the cli.')
    parser.add_argument('--experiment-name', type=str, required=False, default='deept-training',
        help='The name of the experiment this training runs in.')
    parser.add_argument('--use-wandb', type=int, required=False, default=None,
        help='Whether the training shall be logged with wandb.')

    args = parser.parse_args()

    args.resume_training = bool(args.resume_training)
    args.use_wandb = bool(args.use_wandb)

    return vars(args)

def start(config):

    check_and_correct_requested_number_of_gpus(config)

    world_size = max(1, config['number_of_gpus'])

    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(
            setup_and_train,
            args=(config, world_size, ),
            nprocs=world_size
        )
    else:
        if config['do_sweep', False]:
            setup(config, 0, 1, train=True, time=False)
            sweeper = create_sweeper(config)
            sweeper.sweep()
        else:
            setup_and_train(0, config, 1)

def setup_and_train(rank, config, world_size):
    setup(config, rank, world_size, train=True, time=False)
    train(config)

def train(config):
    Context.reset()

    config.print()

    setup_directories(config)
    setup_seeds(config)
    if Settings.use_wandb():
        setup_wandb(config)

    config['update_freq'] = config['update_freq', 1] // Settings.get_number_of_workers()
    my_print(f'Scaled down update_freq to {config["update_freq"]}!')

    train_dataloader, dev_dataloader = create_dataloader(config)
    
    model = create_model(config)
    Context.add_context('model', model)

    criterions = create_scores(config, 'criterions')
    Context.add_context('criterions', criterions)

    scores = create_scores(config, 'scores')
    Context.add_context('scores', scores)

    optimizers, lr_schedulers = create_optimizers_and_lr_schedulers(config)
    Context.add_context('optimizers', optimizers)
    Context.add_context('lr_schedulers', lr_schedulers)

    checkpoint_manager = create_checkpoint_manager(config)

    if config['model_uses_qat', False]:
        from deept.components.quantization import prepare_model_for_qat
        my_print('Warning! Quantization is experimental!')
        Context.overwrite('model', prepare_model_for_qat(config, Context['model']))

    # == No user modifications allowed anymore

    send_to_device()

    if config['number_of_gpus'] > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        Context.overwrite('model', DDP(Context['model'], device_ids=[Settings.get_device()]))

    trainer = Trainer.create_trainer_from_config(config,
        train_dataloader,
        dev_dataloader,
        checkpoint_manager
    )

    my_print('Model:')
    my_print(Context['model'])
    my_print(f'Trainable variables: {sum(p.numel() for p in Context["model"].parameters() if p.requires_grad)}')
    print(f'Worker {Settings.rank()}: Device: {Settings.get_device()}')
    my_print(f'Start training at checkpoint {checkpoint_manager.get_checkpoint_number()}!')
    print_memory_usage()

    result = trainer.train()

    if config['average_last_checkpoints', False]:
        checkpoint_manager.average_last_N_checkpoints(config['checkpoints_to_average'])

    if config['average_last_after_best_checkpoints', False]:
        checkpoint_manager.average_N_after_best_checkpoint(config['checkpoints_to_average'])

    average_time_per_checkpoint_s = checkpoint_manager.checkpoint_duration_accum / checkpoint_manager.checkpoint_count
    my_print(f'Average time per checkpoint: {average_time_per_checkpoint_s:4.2f}s {average_time_per_checkpoint_s/60:4.2f}min')

    if config['use_wandb', False]:
        import wandb
        wandb.run.finish()

    my_print('Done!')

    return result

def create_sweeper(config):
    from deept.sweep import create_sweeper_from_config
    return create_sweeper_from_config(config, train, ())

def create_dataloader(config):
    from deept.data.dataset import create_dataset_from_config
    from deept.data.dataloader import create_dataloader_from_config

    train_dataset = create_dataset_from_config(config, True, 'train_dataset')
    dev_dataset = create_dataset_from_config(config, False, 'dev_dataset')

    train_dataloader = create_dataloader_from_config(config, train_dataset, 
        is_train=True
    )

    dev_dataloader = create_dataloader_from_config(config, dev_dataset, 
        is_train=False
    )

    return train_dataloader, dev_dataloader

def create_model(config):
    from deept.components.model import create_model_from_config
    model = create_model_from_config(config)
    return model

def create_scores(config, key):
    from deept.components.scores import create_score_from_config
    scores = []
    for score_config in config[key]:
        score = create_score_from_config(score_config, config)
        scores.append(score)
    my_print(f'Created {key}: {scores}!')
    return scores

def create_optimizers_and_lr_schedulers(config):
    from deept.components.optimizer import create_optimizers_and_lr_schedulers_from_config
    optimizers, lr_schedulers = create_optimizers_and_lr_schedulers_from_config(config, Context['model'])
    my_print(f'Created optimizers: {optimizers}!')
    my_print(f'Created lr_schedulers: {lr_schedulers}!')
    return optimizers, lr_schedulers

def create_checkpoint_manager(config):
    from deept.utils.checkpoint_manager import CheckpointManager
    checkpoint_manager = CheckpointManager.create_train_checkpoint_manager_from_config(config)
    checkpoint_manager.restore_if_requested()
    my_print(f'Created checkoint_manager!')
    return checkpoint_manager

def send_to_device():
    from deept.utils.globals import Context, Settings

    scores = []
    for score in Context['scores']:
        score = score.to(Settings.get_device()) 
        scores.append(score)

    criterions = []
    for criterion in Context['criterions']:
        criterion = criterion.to(Settings.get_device()) 
        criterions.append(criterion)

    Context.overwrite('scores', scores)
    Context.overwrite('criterions', criterions)
    Context.overwrite('model', Context['model'].to(Settings.get_device()))


if __name__ == '__main__':
    my_print(
""">>=========================================================<<
||                                                         ||
||     $$$$$$$\                             $$$$$$$$\      ||
||     $$  __$$\                            \__$$  __|     ||
||     $$ |  $$ | $$$$$$\   $$$$$$\   $$$$$$\  $$ |        ||
||     $$ |  $$ |$$  __$$\ $$  __$$\ $$  __$$\ $$ |        ||
||     $$ |  $$ |$$$$$$$$ |$$$$$$$$ |$$ /  $$ |$$ |        ||
||     $$ |  $$ |$$   ____|$$   ____|$$ |  $$ |$$ |        ||
||     $$$$$$$  |\$$$$$$$\ \$$$$$$$\ $$$$$$$  |$$ |        ||
||     \_______/  \_______| \_______|$$  ____/ \__|        ||
||                                   $$ |                  ||
||                                   $$ |                  ||
||                                   \__|                  ||
||                                                         ||
>>=========================================================<<""")

    DeepTConfigDescription.create_deept_config_description()

    args = parse_cli_arguments()

    config = Config.parse_config_from_args(args)

    if config['resume_training']:
        config['output_folder'] = config['resume_training_from']

    start(config)