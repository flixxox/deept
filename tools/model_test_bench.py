import _setup_env

import torch

from models import create_model_from_config
from deept.utils.config import Config
from deept.utils.data import Vocabulary, Dataset
from deept.utils.checkpoint_manager import CheckpointManager

# ======== CONFIG

config_file = '/home/fschmidt/code/deept/config/transformer.wmt.en-de.yaml'
checkpoint_path = '/nas/models/neurosys/output/deept/transformer.wmt.en-de/best-so-far/output/checkpoints/ckpt-121.pt'

# ======== CREATION

config = Config.parse_config({'config': config_file})

vocab_src = Vocabulary.create_vocab(config['vocab_src'])
vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

pad_index = vocab_src.PAD

model = create_model_from_config(config, vocab_src, vocab_tgt)

checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config, model)
epoch = checkpoint_manager.restore(checkpoint_path)

# ======== CALLING

model.eval()

src = [
    ['</S>', 'lass', 'sie', 'kog@@', 'i', 'sein', '.', '&quot;', '</S>']
]

tgt = [
    ['</S>', 'let', '&apos;s', 'make']
]

src = [vocab_src.tokenize(s) for s in src]
tgt = [vocab_tgt.tokenize(t) for t in tgt]

print('tokenized src')
for s in src:
    print(s)

print('tokenized tgt')
for t in tgt:
    print(t)

src = torch.Tensor(src).to(torch.int32)
tgt = torch.Tensor(tgt).to(torch.int32)

with torch.no_grad():

    masks, out_mask = model.create_masks(src, tgt, pad_index)
    output, _ = model(src, tgt, **masks)

print(output.numpy())