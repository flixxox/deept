import sre_compile
import _setup_env

import torch
import horovod.torch as hvd

from pytorchmt.util.config import Config
from pytorchmt.util.setup import setup_torch_from_config
from pytorchmt.util.checkpoint_manager import CheckpointManager
from pytorchmt.util.data import Vocabulary, Dataset, BatchGenerator, BucketingBatchAlgorithm, LinearBatchAlgorithm

# ======== CONFIG

config_file = '/home/fschmidt/code/pytorchmt-setups/posnet/configs/relposatt.iwslt.de-en.yaml'

# ======== CREATION

hvd.init()

config = Config.parse_config({'config': config_file})

setup_torch_from_config(config)

vocab_src = Vocabulary.create_vocab(config['vocab_src'])
vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

pad_index = vocab_src.PAD

dataset = Dataset.create_dataset_from_config(config, 'dataset', config['test_src'], config['test_tgt'], vocab_src, vocab_tgt)

batch_generator= BatchGenerator.create_batch_generator_from_config(config, dataset,  BucketingBatchAlgorithm)
    
# ======== CALLING

print('-----')

for i in range(10):

    print('--')

    for _, (src, tgt, out), total_steps in batch_generator.generate_batches():
        
        tgt = tgt.numpy()
        
        for s in tgt:
            s = list(s)
            s = vocab_src.detokenize_list(s)
            s = vocab_src.remove_padding(s)
            s = s[1:-1]
            s = ' '.join(s)

            print(s)