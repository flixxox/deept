import _setup_env

import argparse

import deept.data.dataset
import deept.data.dataloader
import deept.components.model
import deept.components.scores
import deept.components.optimizer
import deept.components.lr_scheduler
from deept.utils.debug import my_print
from deept.utils.setup import import_user_code
from deept.utils.config import DeepTConfigDescription

def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--user-code', type=str, required=False, default=None,
        help="""Supply the directories you would also use during training or search to see
        accurately what modules are available to you.""")

    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':

    my_print(''.center(60, '-'))
    my_print(' Hi! '.center(60, '-'))
    my_print(' Script: print_deept_param_description.py '.center(60, '-'))
    my_print(''.center(60, '-'))

    args = parse_cli_arguments()

    if args['user_code'] is not None:
        import_user_code(args['user_code'])

    DeepTConfigDescription.create_deept_config_description()

    DeepTConfigDescription.print_all_deept_arguments()