
import _setup_env

import argparse

from deept.utils.setup import setup
from deept.utils.config import Config
from deept.utils.debug import my_print
from deept.utils.globals import Context
from deept.utils.setup import import_user_code
from deept.data.datapipe import create_dp_from_config

# ======== CONFIG

config_file = '/home/fschmidt/code/deept-mt/configs/baselines/transformerBase.wmt.en-de.yaml'

# ======== CREATION


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--user-code', type=str, required=False, default=None,
        help="""Supply the directories you would also use during training or search to see
        accurately what modules are available to you.""")

    args = parser.parse_args()

    return vars(args)

def start(config):

    config['output_folder'] = ''
    config['number_of_gpus'] = 1
    config['user_code'] = None
    config['data_dp_overwrite'] = 'webdataset_inspection'

    setup(config, 0, 1, train=False, create_directories=False)

    datapipe = create_dp_from_config(config, 
        config['data_train_root'],
        config['data_train_mask'],
        name='train',
        chunk=False,
        drop_last=True,
        use_max_token_bucketize=True
    )

    num_sentences = 0

    for item in datapipe:
        print(num_sentences, end='\r')
        num_sentences += 1

    print(num_sentences)



if __name__ == '__main__':

    my_print(''.center(60, '-'))
    my_print(' Hi! '.center(60, '-'))
    my_print(' Script: pytorch_data_test_bench.py '.center(60, '-'))
    my_print(''.center(60, '-'))

    args = parse_cli_arguments()

    if args['user_code'] is not None:
        import_user_code(args['user_code'])

    config = Config.parse_config({'config': config_file})

    start(config)

    