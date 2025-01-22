from os.path import join, isdir

import torch
import numpy as np

from deept.utils.debug import my_print
from deept.utils.globals import Settings
from deept.utils.config import DeepTConfigDescription


def check_and_correct_requested_number_of_gpus(config, train=True):

    num_gpus_avail = torch.cuda.device_count()

    my_print(f'Number of GPUs available: {num_gpus_avail}')
    my_print('Available devices:', [torch.cuda.get_device_name(i) for i in range(num_gpus_avail)])

    config['number_of_gpus'] = max(0, config['number_of_gpus'])

    assert config['number_of_gpus'] <= num_gpus_avail, f"""Not enough GPUs available! 
        Avail: {num_gpus_avail}, Requested {config["number_of_gpus"]}"""

    if not train: # We do not support multi-gpu for search
        config['number_of_gpus'] = min(1, config['number_of_gpus'])

    my_print(f'Requested number of GPUs after check: {config["number_of_gpus"]}')

def setup(config, rank, world_size, 
    train=True,
    time=False
):
    if config['user_code'] is not None:
        import_user_code(config['user_code'])
    DeepTConfigDescription.create_deept_config_description()
    setup_settings(config, rank, world_size, train, time)
    setup_torch(config)
    if config['number_of_gpus'] > 0:
        setup_cuda(config)
    if config['number_of_gpus'] > 1:
        setup_ddp(config)

def import_user_code(paths_to_user_code):
    import sys
    import importlib
    from os import listdir
    from os.path import isdir, join

    def __import_file(module):
        if module.startswith('.'):
            module = module[1:]
        if module.endswith('.'):
            module = module[:-1]
        module = importlib.import_module(module)

    def __can_be_imported(filename):
        return (filename.endswith('.py')
            and not filename.startswith('.')
            and not filename.startswith('_'))

    def __has_init_file(path):
        for filename in listdir(path):
            if filename == '__init__.py':
                return True
        return False

    def __import_files_recursive(prefix, path):
        dir_has_init = __has_init_file(path)
        for filename in listdir(path):
            filepath = join(path, filename)
            if isdir(filepath) and not filename.startswith('.') and not filename.startswith('_'):
                __import_files_recursive(prefix + f'.{filename}', filepath)
            else:
                if dir_has_init and __can_be_imported(filename):
                    __import_file(prefix + f'.{filename.replace(".py", "")}')
    
    if not isinstance(paths_to_user_code, list):
        paths_to_user_code = [paths_to_user_code]

    for path in paths_to_user_code:

        if not isdir(path):
            raise ValueError(f'Error! User code directory not found: {path}!')

        if path.endswith('/'):
            path = path[:-1]

        __import_files_recursive('', path)

def setup_settings(config, rank, world_size, train, time):
    Settings.set_rank(rank)
    Settings.set_number_of_workers(world_size)
    Settings.set_train_flag(train)
    Settings.set_time_flag(time)
    Settings.set_use_wandb(config['use_wandb', False])
    if config['number_of_gpus'] < 1:
        my_print('Limiting to CPU!')
        Settings.set_cpu()
        Settings.set_device('cpu')
    else:
        Settings.set_device(f'cuda:{Settings.rank()}')

def setup_directories(config):
    def __maybe_create_dir(dir):
        from os import mkdir
        if not isdir(dir):
            mkdir(dir)
    
    Settings.reset_directories()
    if config.has_key('output_folder_root'):
        Settings.add_dir('output_dir_root', config['output_folder_root'])
    Settings.add_dir('output_dir', config['output_folder'])
    Settings.add_dir('numbers_dir', join(config['output_folder'], 'numbers'))
    Settings.add_dir('checkpoint_dir',  join(config['output_folder'], 'checkpoints'))

    if not Settings.is_training():
        Settings.add_dir('search_dir', join(config['output_folder'], 'search'))
    
    if Settings.rank() == 0:
        if Settings.is_training():
            if Settings.has_dir('output_dir_root'):
                __maybe_create_dir(Settings.get_dir('output_dir_root'))
            __maybe_create_dir(Settings.get_dir('output_dir'))
            __maybe_create_dir(Settings.get_dir('checkpoint_dir'))
            __maybe_create_dir(Settings.get_dir('numbers_dir'))

            assert isdir(Settings.get_dir('checkpoint_dir')), f"""Something went wrong in creating directories! 
                Expected to have the directory {Settings.get_dir('checkpoint_dir')}"""

            assert isdir(Settings.get_dir('numbers_dir')), f"""Something went wrong in creating directories! 
                Expected to have the directory {Settings.get_dir('numbers_dir')}"""

        else:
            __maybe_create_dir(Settings.get_dir('search_dir'))

            assert isdir(Settings.get_dir('search_dir')), f"""Something went wrong in creating directories! 
                Expected to have the directory {Settings.get_dir('search_dir')}"""

def setup_torch(config):
    if config['deterministic', False]:
        torch.use_deterministic_algorithms(True,
            warn_only=config['deterministic_warn_only', False]
        )

    if not Settings.is_training() and config['quantize_post_training', False]:
        backend = config['quantize_backend', 'x86']
        if backend == 'x86':
            my_print('Using quantization backend x86!')
            torch.backends.quantized.engine = 'x86'
        elif backend == 'qnnpack':
            my_print('Using quantization backend qnnpack!')
            torch.backends.quantized.engine = 'qnnpack'
        else:
            raise ValueError(f'Unrecognized quantization backend: {backend}! Accepted values ["x86", "qnnpack"].')

    if config['torch_disable_detect_anomaly', False]:
        torch.autograd.set_detect_anomaly(False)

    torch.set_printoptions(precision=4, sci_mode=False)

def setup_cuda(config):
    torch.cuda.set_device(Settings.get_device())
    if config['deterministic', False]:
        torch.backends.cudnn.deterministic = True
    if config['disable_cudnn', False]: 
        torch.backends.cudnn.enabled = False
    if config['disable_cudnn_benchmark_mode', False]:
        torch.backends.cudnn.benchmark = False

def setup_ddp(config):
    import os
    import torch.distributed as dist

    my_print('Setting up distributed training!')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        rank=Settings.rank(),
        world_size=Settings.get_number_of_workers()
    ) # Uses nccl for gpu and gloo for cpu communication

def setup_seeds(config):
    import random
    Settings.set_global_seed(config['seed', 0])
    seed = Settings.get_global_seed()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if Settings.is_gpu():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_wandb(config):
    import wandb
    wandb.init(
        config=config.asdict(),
        dir=Settings.get_dir('output_dir'),
        job_type='train',
        mode=config['wandb_mode', 'offline'],
        project=config['wandb_project'],
        name=config['experiment_name', None]
    )