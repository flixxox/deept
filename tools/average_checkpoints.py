import _setup_env
from os.path import join

import torch
import numpy as np
import horovod.torch as hvd

from models import create_model_from_config
from pytorchmt.util.config import Config
from pytorchmt.util.data import Vocabulary
from pytorchmt.util.setup import setup_torch_from_config
from pytorchmt.util.checkpoint_manager import CheckpointManager

# ======== CONFIG

topic = 'posnet'
model = 'posnet'
task = 'iwslt.de-en'
train_dir = '/nas/models/neurosys/output/posnet/posnet.iwslt.de-en/rPosNet-glu-noNormalize'
checkpoints_to_average = np.arange(31, 61)

config_file = f'{train_dir}/pytorchmt-setups/{topic}/configs/{model}.{task}.yaml'
checkpoint_dir = f'{train_dir}/output/checkpoints'

# ======== CREATION

hvd.init()

config = Config.parse_config({'config': config_file})

config['checkpoint_dir'] = checkpoint_dir
config['number_of_gpus'] = 0

setup_torch_from_config(config)

vocab_src = Vocabulary.create_vocab(config['vocab_src'])
vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

pad_index = vocab_src.PAD

checkpoint_manager = CheckpointManager.create_train_checkpoint_manager_from_config(config, None, None)

# ======== CALLING

checkpoint_paths = [join(checkpoint_dir, f'ckpt-{i}.pt') for i in checkpoints_to_average]

checkpoint_manager.average_checkpoints(checkpoint_paths)