
import _setup_env

from deept.util.setup import setup
from deept.util.config import Config
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
    config['data_train_mask']
)

dataloader = create_dataloader_from_config(config,
    datapipe, 
    shuffle=True
)


for item in dataloader:
    print('==')
    print('out', item['out'].shape)

dataloader.shutdown()