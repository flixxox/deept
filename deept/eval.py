
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
        help='The folder in which to write the evaluation numbers.')
    parser.add_argument('--load-ckpt-from', type=str, required=False, default='', 
        help='The checkpoint to evaluate.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None,
        help='This is usually specified in the config but can also be overwritten from the cli.')
    parser.add_argument('--use-wandb', type=int, required=False, default=None,
        help='Whether the evaluation shall be logged with wandb.')

    args = parser.parse_args()

    args.use_wandb = bool(args.use_wandb)

    return vars(args)

def eval(config):
    assert config['number_of_gpus'] <= 1, 'Too many GPUs specified. For evaluation, only one is supported.'

    setup(config, 0, 1, train=False, time=False)

    config['load_weights'] = True
    config['load_weights_from'] = config['load_ckpt_from']
    config['experiment_name'] = f'eval_{config["model"]}'

    Context.reset()

    config.print()

    setup_directories(config)
    setup_seeds(config)
    if Settings.use_wandb():
        setup_wandb(config)

    test_dataloader = create_dataloader(config)
    
    model = create_model(config)
    Context.add_context('model', model)

    scores = create_scores(config, 'scores')
    Context.add_context('scores', scores)

    criterions = create_scores(config, 'criterions')
    Context.add_context('criterions', criterions)

    checkpoint_manager = create_checkpoint_manager(config)

    # == No user modifications allowed anymore

    send_to_device()

    trainer = Trainer.create_eval_trainer_from_config(config,
        test_dataloader,
        checkpoint_manager
    )

    my_print('Model:')
    my_print(Context['model'])
    my_print(f'Trainable variables: {sum(p.numel() for p in Context["model"].parameters() if p.requires_grad)}')
    print(f'Worker {Settings.rank()}: Device: {Settings.get_device()}')
    my_print(f'Evaluating checkpoint {checkpoint_manager.get_checkpoint_number()}!')
    print_memory_usage()

    result = trainer.eval()

    if config['use_wandb', False]:
        import wandb
        wandb.run.finish()

    my_print('Done!')

def create_sweeper(config):
    from deept.sweep import create_sweeper_from_config
    return create_sweeper_from_config(config, train, ())

def create_dataloader(config):
    from deept.data.dataset import create_dataset_from_config
    from deept.data.dataloader import create_dataloader_from_config

    test_dataset = create_dataset_from_config(config, False, 'test', 'test_dataset')

    test_dataloader = create_dataloader_from_config(config, test_dataset, 
        is_train=False
    )

    return test_dataloader

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

def create_checkpoint_manager(config):
    from deept.utils.checkpoint_manager import CheckpointManager
    checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config)
    #checkpoint_manager.restore_if_requested()
    my_print(f'Created checkoint_manager!')
    return checkpoint_manager

def send_to_device():
    from deept.utils.globals import Context, Settings

    scores = []
    for score in Context['scores']:
        score = score.to(Settings.get_device()) 
        scores.append(score)

    Context.overwrite('scores', scores)
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

    eval(config)