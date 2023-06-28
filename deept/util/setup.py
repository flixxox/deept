
import torch
import torch.distributed as dist

from deept.util.debug import my_print
from deept.util.globals import Globals


def check_devices(config):
    """ 
    This function is run before processes are spawn.
    It checks number_of_gpus and sets Globals.number_of_workers.
    """

    num_gpus_avail = torch.cuda.device_count()

    my_print(f'Number of GPUs available: {num_gpus_avail}')
    my_print('Available devices:', [torch.cuda.get_device_name(i) for i in range(num_gpus_avail)])

    config['number_of_gpus'] = max(0, config['number_of_gpus'])

    assert config['number_of_gpus'] <= num_gpus_avail, f'Not enough GPUs available! Avail: {num_gpus_avail}, Requested {config["number_of_gpus"]}'

    if not Globals.is_training(): # We do not support multi-gpu for search
        config['number_of_gpus'] = min(1, config['number_of_gpus'])

    Globals.set_number_of_workers(max(1, config['number_of_gpus']))

def setup_torch(config):

    if config['deterministic', False]:
        torch.use_deterministic_algorithms(True)

    if Globals.get_number_of_workers() <= 0:
        my_print('Limiting to CPU!')
        Globals.set_cpu()
        Globals.set_device('cpu')

    else:

        setup_ddp(config)

def setup_ddp(config):

    import os

    my_print('Setting up DDP!')

    config['update_freq'] = config['update_freq'] // Globals.get_number_of_workers()

    my_print(f'Scaled down update_freq to {config["update_freq"]}!')

    Globals.set_device(f'cuda:{Globals.rank()}')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", rank=Globals.rank(), world_size=Globals.get_number_of_workers())