
import _setup_env

from deept.util.setup import setup
from deept.util.config import Config
from deept.util.data import create_dp_from_config

# ======== CONFIG

config_file = '/home/fschmidt/code/deept-mt/configs/baselines/transformer.iwslt.de-en.yaml'

# ======== CREATION

config = Config.parse_config({'config': config_file})

config['output_folder'] = ''
config['number_of_gpus'] = 0
config['user_code'] = None

setup(config, 0, 1, train=False, create_directories=False)

train_datapipe = create_dp_from_config(config,
    config['data_train_root'],
    config['data_train_mask'],
    bucket_batch=True
)

# rs = MultiProcessingReadingService(num_workers=1)
# dl = DataLoader2(pipe, reading_service=rs)


for item in train_datapipe:
    print('==')
    print('out', item['out'].shape)
