
import _setup_env

from deept.util.setup import setup
from deept.util.config import Config
from deept.util.globals import Context
from deept.util.datapipe import create_dp_from_config
from deept.util.dataloader import create_dataloader_from_config

# ======== CONFIG

config_file = '/home/fschmidt/code/deept-mt/configs/baselines/transformer.iwslt.de-en.yaml'

# ======== CREATION

config = Config.parse_config({'config': config_file})

config['output_folder'] = ''
config['number_of_gpus'] = 1
config['user_code'] = None

setup(config, 0, 1, train=False, create_directories=False)

datapipe = create_dp_from_config(config, 
    config['data_train_root'],
    config['data_train_mask'],
    name='train',
    chunk=False
)


dataloader = create_dataloader_from_config(config,
    datapipe, 
    shuffle=True
)

voacb_tgt = Context['vocab_tgt']

for item in dataloader:
    assert (
        item['src'].shape[1] <= config['max_sample_size'] and 
        item['tgt'].shape[1] <= config['max_sample_size'] and
        item['out'].shape[1] <= config['max_sample_size']), (f"""Error! Exceeded sentence length! 
            src {item['src'].shape} tgt {item['src'].shape}!""")

    print('===')
    print(item['tgt'].shape)

    num_tokens = item['tgt'].shape[0] * item['tgt'].shape[1]
    num_pad_tokens = (item['tgt'] != voacb_tgt.PAD).sum()

    print(f'Batch effectiveness: {num_pad_tokens/num_tokens:4.2f}')

dataloader.shutdown()