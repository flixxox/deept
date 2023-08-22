
import _setup_env

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as torchquant


import argparse

from deept.util.config import Config
from deept.util.debug import my_print
from deept.util.globals import Settings
from deept.util.setup import (
    setup,
    check_and_correct_requested_number_of_gpus
)


from deept_mt.quantization.quant_utils import create_activation_quantizer
from deept_mt.quantization.quantized_modules import (
    QuantizedLinear,
    QuantizedLinearRelu
)

# ======== CONFIG

config_file = '/home/fschmidt/code/deept-mt/configs/baselines/transformer.iwslt.de-en.yaml'

# ======== CREATION


class TestModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        self.quantizer = create_activation_quantizer(4, 'MinMax')
        self.linear = QuantizedLinear(5, 10, 4)

    def __call__(self, x):
        x = self.quantizer(x)
        x = self.linear(x)
        return x


def start(config):

    config['output_folder'] = ''
    config['number_of_gpus'] = 1
    config['user_code'] = None

    check_and_correct_requested_number_of_gpus(config)
    setup(config, 0, 1, train=False, create_directories=False)

    torch.set_printoptions(sci_mode=False)

    test_module = TestModule().to(Settings.get_device())
    x = (torch.rand(5,5).to(Settings.get_device()) * 100) - 50

    y = test_module(x)

    y.sum().backward()


if __name__ == '__main__':

    my_print(''.center(60, '-'))
    my_print(' Hi! '.center(60, '-'))
    my_print(' Script: quant_test.py '.center(60, '-'))
    my_print(''.center(60, '-'))

    config = Config.parse_config({'config': config_file})

    start(config)

    