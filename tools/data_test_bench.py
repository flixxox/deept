
import _setup_env

import argparse


from deept.util.setup import setup
from deept.util.config import Config
from deept.util.debug import my_print
from deept.util.globals import Context
from deept.util.setup import import_user_code
from deept.util.debug import print_memory_usage
from deept.data.datapipe import create_dp_from_config
from deept.data.dataloader import create_dataloader_from_config

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

    setup(config, 0, 1, train=False, create_directories=False)

    datapipe = create_dp_from_config(config, 
        config['data_train_root'],
        config['data_train_mask'],
        name='train',
        chunk=False,
        drop_last=True,
        use_max_token_bucketize=True
    )

    dataloader = create_dataloader_from_config(config,
        datapipe,
        shuffle=True
    )

    voacb_tgt = Context['vocab_tgt']

    effectiveness_accum = 0
    steps = 0
    num_sentences = 0

    for item in dataloader:

        assert (
            item['src'].shape[1] <= config['max_sample_size'] and 
            item['tgt'].shape[1] <= config['max_sample_size'] and
            item['out'].shape[1] <= config['max_sample_size']), (f"""Error! Exceeded sentence length! 
                src {item['src'].shape} tgt {item['src'].shape}!""")

        my_print('========')

        num_tokens = item['tgt'].shape[0] * item['tgt'].shape[1]
        num_pad_tokens = (item['tgt'] != voacb_tgt.PAD).sum()

        cur_effectiveness = (num_pad_tokens/num_tokens)

        my_print(f'Batch effectiveness: {cur_effectiveness:4.2f}')
        my_print(f'Target length: {item["tgt"].shape[1]}')
        print_memory_usage()

        effectiveness_accum += cur_effectiveness
        steps += 1
        num_sentences += item['tgt'].shape[0]

    my_print(f'Average effectiveness: {(effectiveness_accum/steps):4.2f}!')
    my_print(f'Steps: {(steps)}!')
    my_print(f'Num Sentences: {(num_sentences)}!')

    dataloader.shutdown()


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

    