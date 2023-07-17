
import torch

from deept.util.debug import my_print
from deept.util.globals import Settings


def import_user_code(path_to_user_code):

    import importlib
    from os import listdir
    from os.path import isdir

    assert isdir(path_to_user_code)

    for file in listdir(path_to_user_code):
        if file.endswith('.py') and not file.startswith('_'):
            module_name = file[:file.find('.py')]
            module = importlib.import_module('deept_user.' + module_name)

def check_and_correct_requested_number_of_gpus(config, train=True):

    num_gpus_avail = torch.cuda.device_count()

    my_print(f'Number of GPUs available: {num_gpus_avail}')
    my_print('Available devices:', [torch.cuda.get_device_name(i) for i in range(num_gpus_avail)])

    config['number_of_gpus'] = max(0, config['number_of_gpus'])

    assert config['number_of_gpus'] <= num_gpus_avail, f'Not enough GPUs available! Avail: {num_gpus_avail}, Requested {config["number_of_gpus"]}'

    if not train: # We do not support multi-gpu for search
        config['number_of_gpus'] = min(1, config['number_of_gpus'])

    my_print(f'Requested number of GPUs after check: {config["number_of_gpus"]}')

def setup(config, rank, world_size, train=True, time=False):
    setup_settings(config, rank, world_size, train, time)
    setup_torch(config)
    if config['number_of_gpus'] > 0:
        setup_ddp(config)

def setup_settings(config, rank, world_size, train, time):

    setup_directories(config)

    Settings.set_rank(rank)
    Settings.set_number_of_workers(world_size)
    Settings.set_train_flag(train)
    Settings.set_time_flag(time)
    Settings.set_global_seed(config['seed', 80420])
    if config['number_of_gpus'] < 1:
        my_print('Limiting to CPU!')
        Settings.set_cpu()
        Settings.set_device('cpu')
    else:
        Settings.set_device(f'cuda:{Settings.rank()}')

def setup_directories(config):
    from os.path import join

    Settings.add_dir('output_dir', config['output_folder'])
    Settings.add_dir('numbers_dir', join(config['output_folder'], 'numbers'))
    Settings.add_dir('checkpoint_dir',  join(config['output_folder'], 'checkpoints'))

    __maybe_create_dir(Settings.get_dir('checkpoint_dir'))
    __maybe_create_dir(Settings.get_dir('numbers_dir'))

def __maybe_create_dir(dir):
    from os import mkdir
    from os.path import isdir
    if not isdir(dir) and Settings.rank() == 0:
        mkdir(checkpoint_dir)

def setup_torch(config):
    if config['deterministic', False]:
        torch.use_deterministic_algorithms(True)

def setup_ddp(config):

    import os
    import torch.distributed as dist

    my_print('Setting up distributed training!')

    config['update_freq'] = config['update_freq'] // Settings.get_number_of_workers()
    my_print(f'Scaled down update_freq to {config["update_freq"]}!')

    torch.cuda.set_device(Settings.get_device())

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(rank=Settings.rank(), world_size=Settings.get_number_of_workers()) # Uses nccl for gpu and gloo for cpu communication