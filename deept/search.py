
import argparse

import torch

from deept.search.seeker import Seeker
from deept.util.globals import Settings, Context
from deept.data.datapipe import create_dp_from_config
from deept.model.model import create_model_from_config
from deept.util.checkpoint_manager import CheckpointManager
from deept.data.dataloader import create_dataloader_from_config
from deept.util.debug import my_print, get_number_of_trainable_variables
from deept.search.search_algorithm import create_search_algorithm_from_config
from deept.util.config import (
    Config,
    DeepTConfigDescription
)
from deept.util.setup import (
    setup,
    check_and_correct_requested_number_of_gpus
)


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help="""The path to the config.yaml which contains all user defined parameters. 
            It may or may not match the one trained with. This is up to the user to ensure.""")
    parser.add_argument('--user-code', type=str, nargs='+', required=False, default=None,
        help="""One or multiple paths to directories containing user code.""")
    parser.add_argument('--checkpoint-path', type=str, required=True, 
        help='The checkpoint.pt file containing the model weights.')
    parser.add_argument('--output-folder', type=str, required=False, default=None, 
        help='The output folder in which to write the score and hypotheses.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None, 
        help="""This is usually specified in the config but can also be overwritten from the cli. 
            However, in search this can only be 0 or 1. We do not support multi-gpu decoding.
            If you set it to >1 we will set it back to 1 so that you dont need to modify the config in search.""")

    args = parser.parse_args()

    return vars(args)

def search(config):

    check_and_correct_requested_number_of_gpus(config, 
        train=False)

    setup(config, 0, 1, train=False)

    config['batch_size'] = config['batch_size_search']

    if config['search_test_set', False]:
        data_root = config['data_test_root']
        data_mask = config['data_test_mask']
        corpus_size = config['corpus_size_test']
    else:
        data_root = config['data_dev_root']
        data_mask = config['data_dev_mask']
        corpus_size = config['corpus_size_dev']

    datapipe = create_dp_from_config(config,
        data_root,
        data_mask,
        name='search',
        chunk=False,
        drop_last=False,
        use_max_token_bucketize=False,
    )

    dataloader = create_dataloader_from_config(config, datapipe, 
        shuffle=False,
        num_worker_overwrite=1
    )

    model = create_model_from_config(config)
    Context.add_context('model', model)

    checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config)
    checkpoint_manager.restore(config['checkpoint_path'])

    my_print(f'Trainable variables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if config['quantize_post_training', False]:
        from deept.model.quantization import PostTrainingQuantizer
        quantizer = PostTrainingQuantizer.create_from_config(config)
        Context.overwrite('model', quantizer.quantize(Context['model']))

    search_algorithm = create_search_algorithm_from_config(config)

    seeker = Seeker.create_from_config(config,
        dataloader,
        search_algorithm,
        checkpoint_manager.checkpoint_count,
        corpus_size
    )

    my_print('Model:', Context['model'])

    my_print(f'Searching checkpoint {checkpoint_manager.checkpoint_count}!')

    model.eval()

    with torch.no_grad():
        
        seeker.search()
    
    dataloader.shutdown()

    my_print('Done!')


if __name__ == '__main__':

    my_print(''.center(60, '-'))
    my_print(' Hi! '.center(60, '-'))
    my_print(' Script: search.py '.center(60, '-'))
    my_print(''.center(60, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)

    config.print_config()

    search(config)