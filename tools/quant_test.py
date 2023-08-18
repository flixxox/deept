
import _setup_env

import torch

import argparse

from deept.util.setup import setup
from deept.util.config import Config
from deept.util.debug import my_print

# ======== CONFIG

config_file = '/home/fschmidt/code/deept-mt/configs/baselines/transformer.iwslt.de-en.yaml'

# ======== CREATION


def start(config):

    config['output_folder'] = ''
    config['number_of_gpus'] = 0
    config['user_code'] = None

    setup(config, 0, 1, train=False, create_directories=False)

    torch.set_printoptions(sci_mode=False)

    fake_quant = torch.ao.quantization.fake_quantize.FakeQuantize(
        quant_min=-4,
        quant_max=4,
        dtype=torch.qint8
    )

    quant_stub = torch.ao.quantization.QuantStub(
        qconfig =  torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
            weight=torch.ao.quantization.default_observer.with_args(dtype=torch.qint8)
        )
    )

    i = 0

    while True:

        x = torch.rand(5,5) * 100

        my_print('Before', x, x.dtype)

        x_q = fake_quant(x)

        x_qq = quant_stub(x)

        my_print('After', x_q, x_q.dtype)
        my_print('After', x_qq, x_qq.dtype)

        if i % 1500 == 0:
            my_print('-- Diff', torch.sum(torch.abs(x-x_q))/25)
            my_print('Scale', fake_quant.scale)
            my_print('ZP', fake_quant.zero_point)

        break

        i += 1


if __name__ == '__main__':

    my_print(''.center(60, '-'))
    my_print(' Hi! '.center(60, '-'))
    my_print(' Script: quant_test.py '.center(60, '-'))
    my_print(''.center(60, '-'))

    config = Config.parse_config({'config': config_file})

    start(config)

    