import os
import gc
import psutil

import torch
from resource import getrusage, RUSAGE_SELF

from deept.utils.globals import Settings

def my_print(*args, **kwargs):
    if Settings.rank() == 0 or Settings.rank() == None:
        print(*args, flush=True, **kwargs)

def get_number_of_trainable_variables(model):
    sum = 0
    for var in model.trainable_variables:
        cur = 1
        for s in var.shape:
            cur *= s
        sum += cur
    return sum

def print_memory_usage():
    try:
        current_ram = psutil.Process(os.getpid()).memory_info()[0]/2.**30
        peak_ram = getrusage(RUSAGE_SELF).ru_maxrss/10.**6
    except ImportError:
        my_print('Warning! Please install psutil to print RAM usage.')
        current_ram = 0
        peak_ram = 0

    if Settings.is_gpu():
        current_gpu_memory = torch.cuda.memory_allocated(device=Settings.get_device())
        peak_gpu_memory = torch.cuda.max_memory_allocated(device=Settings.get_device())
    else:
        current_gpu_memory = 0
        peak_gpu_memory = 0

    print(f'Worker {Settings.rank()}: Memory [Current/Peak]. GPU: [{(current_gpu_memory / 1e9):4.2f}GB/{(peak_gpu_memory / 1e9):4.2f}GB], RAM: [{current_ram:4.2f}GB/{peak_ram:4.2f}GB]', flush=True)

def print_allocated_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                my_print(type(obj), obj.size())
        except:
            pass

def search_name_of_parameter(model, param):
    names = []
    for name, p in model.named_parameters():
        if torch.equal(p, param):
            names.append(name)
    return names