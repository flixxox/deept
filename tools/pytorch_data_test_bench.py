
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

setup(config, 0, 1, train=False)

train_datapipe = create_dp_from_config(config,
    config['data_train_root'],
    config['data_train_mask'],
    bucket_batch=True
)

for item in train_datapipe:
    print(item)

# def decode(item):
#     key, value = item
#     return key, value.read().decode('utf-8')

# def my_len_fn(data):
#     idx, src, tgt = data
#     return len(tgt)

# def normalize_names(item):
#     assert isinstance(item, dict)
#     dict_norm = {}
#     for k, v in item.items():
#         if 'source' in k or 'src' in k:
#             dict_norm['src'] = v
#         elif 'target' in k or 'tgt' in k:
#             dict_norm['tgt'] = v
#         else:
#             dict_norm[k] = v
#     return dict_norm

# def to_list(item):
#     src = item['src'].split(' ')
#     tgt = item['tgt'].split(' ')
#     return item['__key__'], src, tgt

# def tensorize(batch):

#     #lens = [max([len(sample[i]) for sample in batch]) for i in range(len(batch))]
    
#     return batch


# General pipeline layout:
# 1. Generic: Load binary stream
# 2. User specific: Decode (text, images, audio etc.)
# 3. Generic: to webdataset -> sharding
# 4. User specific: pre-processing
# 5. Generic with user specified len_fn: Batching
# 6. User specific: Collate

# pipe = (
#     dp.iter.FileLister(root='/home/fschmidt/data/iwslt/de-en/webdataset', masks='train.tar', recursive=False, abspath=True)
#     .shuffle(buffer_size=10000) # shuffle shards
#     .open_files(mode="b")
#     .load_from_tar()
#     .map(decode)
#     .webdataset()
#     .shuffle(buffer_size=10000) # shuffle shards
#     .sharding_filter() # Distributes across processes
#     .map(normalize_names)
#     .map(to_list)
#     .max_token_bucketize(max_token_count=100, len_fn=my_len_fn, include_padding=False)
#     .shuffle(buffer_size=30)
#     .map(tensorize) # use the collate functionality for this
# )

# rs = MultiProcessingReadingService(num_workers=1)
# dl = DataLoader2(pipe, reading_service=rs)

# sum_len = 0
# for x in dl:
#     print(x)