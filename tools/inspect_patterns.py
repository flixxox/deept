from os.path import join

import torch
import numpy as np
import horovod.torch as hvd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models import create_model_from_config
from pytorchmt.util.config import Config
from pytorchmt.util.data import Vocabulary, Dataset
from pytorchmt.util.setup import setup_torch_from_config
from pytorchmt.util.checkpoint_manager import CheckpointManager

# ======== CONFIG

config_file = '/home/fschmidt/code/pytorchmt-setups/posnet/configs/gaussianattnBase.wmt.en-de.yaml'
checkpoint_path = '/nas/models/neurosys/output/posnet/gaussianattnBase.wmt.en-de/enc-dec-std1/output/checkpoints/ckpt-avg-last.pt'

# ======== CREATION

hvd.init()

config = Config.parse_config({'config': config_file})

config.print_config()

config['number_of_gpus'] = 0
config['load_weights'] = False
config['resume_training'] = False
config['stepwise'] = False

setup_torch_from_config(config)

vocab_src = Vocabulary.create_vocab(config['vocab_src'])
vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

pad_index = vocab_src.PAD

model = create_model_from_config(config, vocab_src, vocab_tgt)

checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config, model)
checkpoint_manager.restore(checkpoint_path)

# ======== FUNCTIONS

def plot(p_enc, p_src, p_tgt, orig_src, orig_tgt, dest='/home/fschmidt/3_plot.png'):

    p_enc = p_enc.squeeze().numpy()
    p_src = p_src.squeeze().numpy()
    p_tgt = p_tgt.squeeze().numpy()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,8))

    fig.suptitle('Patterns En->De', fontsize=30)

    titles = []
    titles.append('Encoder Pattern')
    titles.append('Source Pattern')
    titles.append('Target Pattern')
    
    ims = []
    ims.append(axes[0].imshow(p_enc, cmap=plt.cm.Blues))
    ims.append(axes[1].imshow(p_src, cmap=plt.cm.Blues))
    ims.append(axes[2].imshow(p_tgt, cmap=plt.cm.Blues))

    x_ticks = []
    x_ticks.append(orig_src)
    x_ticks.append(orig_src)
    x_ticks.append(orig_tgt[1:] + ['</S>'])

    y_ticks = []
    y_ticks.append(orig_src)
    y_ticks.append(orig_tgt[1:] + ['</S>'])
    y_ticks.append(orig_tgt[1:] + ['</S>'])

    for title, im, ax, xs, ys in zip(titles, ims, axes, x_ticks, y_ticks):

        ax.titlesize = 40
        ax.title.set_text(title)

        ax.set_xticks(np.arange(len(xs)), xs, rotation=90, fontsize=15)
        ax.set_yticks(np.arange(len(ys)), ys, rotation=0, fontsize=15)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, fraction=1, cax=cax)
    
    plt.tight_layout()
    fig.savefig(dest)

def plot_all_matrices(matrices, orig_src=None, orig_tgt=None, name='encoder_self', dest='/home/fschmidt', exclude_eos=True):

    fig, axes = plt.subplots(nrows=config['encL'], ncols=config['nHeads'], figsize=(100,100))

    fig.suptitle(f'Matrices {name}', fontsize=50)

    for l in range(config['encL']):
        for h in range(config['nHeads']):
            
            if exclude_eos:
                im = axes[l][h].imshow(matrices[l][0,h,1:,1:], cmap=plt.cm.Blues)
            else:
                im = axes[l][h].imshow(matrices[l][0,h,:,:], cmap=plt.cm.Blues)

            axes[l][h].title.set_text(f'L={l} H={h}')
            axes[l][h].title.set_fontsize(30)
            
            if orig_src is not None:
                axes[l][h].set_xticks(np.arange(len(orig_src)), orig_src, rotation=90, fontsize=30)
            if orig_tgt is not None:
                axes[l][h].set_yticks(np.arange(len(orig_tgt)), orig_tgt, rotation=0, fontsize=30)

            divider = make_axes_locatable(axes[l][h])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, fraction=1, cax=cax)
    
    fig.tight_layout()
    fig.savefig(f'{dest}/{name}.png')

def export(matrices, dest='/nas/models/neurosys/output/patternMatching/reference-patterns'):
    torch.save({
        'matrices': matrices
    }, join(dest, 'matrices.ckpt'))

def calculate_patterns(model, matrices):
        
        eos_scale = .1
        L_src = matrices['encoder/layer0/self_att']['value'].shape[-1]
        L_tgt = matrices['decoder/layer0/self_att']['value'].shape[-1]
        I_src = torch.eye(L_src)
        I_tgt = torch.eye(L_tgt)
        mask = torch.tril(torch.ones((L_tgt, L_tgt)))
        mask = (mask == 0).unsqueeze(0).unsqueeze(0)

        # Encoder Pattern
        p_enc = matrices['encoder/layer0/self_att']['value']
        p_enc = torch.mean(p_enc, 1)
        p_enc = 0.5*I_src + 0.5*p_enc
        p_enc[:,:,0] *= eos_scale
        p_enc[:,:,-1] *= eos_scale
        for l in range(1, model.encL):
            A_l = matrices[f'encoder/layer{l}/self_att']['value']
            A_l = torch.mean(A_l, 1)
            A_l = 0.5*I_src + 0.5*A_l
            A_l[:,:,0] *= eos_scale
            A_l[:,:,-1] *= eos_scale
            p_enc = torch.matmul(A_l, p_enc)

        # Source Pattern
        C_0 = matrices['decoder/layer0/cross_att']['value']
        C_0 = torch.mean(C_0, 1)
        p_src = torch.matmul(C_0, p_enc)
        for l in range(1, model.decL):
            B_l = matrices[f'decoder/layer{l}/self_att']['value']
            B_l = torch.mean(B_l, 1)
            B_l[:,:,0] *= eos_scale
            C_l = torch.mean(matrices[f'decoder/layer{l}/cross_att']['value'], 1)
            p_src =0.5*torch.matmul(B_l, p_src) + 0.5*torch.matmul(C_l, p_enc)

        # Target Pattern
        p_tgt = matrices['decoder/layer0/self_att']['value']
        p_tgt = p_tgt.masked_fill(mask, 0.)
        p_tgt = torch.mean(p_tgt, 1)
        p_tgt = 0.5*I_tgt + 0.5*p_tgt
        p_tgt[:,:,0] *= eos_scale
        for l in range(1, model.decL):
            B_l = matrices[f'decoder/layer{l}/self_att']['value']
            B_l = B_l.masked_fill(mask, 0.)
            B_l = torch.mean(B_l, 1)
            B_l = 0.5*I_tgt + 0.5*B_l
            B_l[:,:,0] *= eos_scale
            p_tgt = torch.matmul(B_l, p_tgt)

        return p_enc, p_src, p_tgt

def summarize_matrices(model, matrices):

    A = matrices['encoder/layer0/self_att']['value'].sum(1)
    B = matrices['decoder/layer0/self_att']['value'].sum(1)
    C = matrices['decoder/layer0/cross_att']['value'].sum(1)

    for l in range(model.encL):
        A += matrices[f'encoder/layer{l}/self_att']['value'].sum(1)

    for l in range(model.decL):
        B += matrices[f'decoder/layer{l}/self_att']['value'].sum(1)
        C += matrices[f'decoder/layer{l}/cross_att']['value'].sum(1)

    return A, B, C


# ======== SCRIPT

model.eval()

# orig_src = [
#     ['</S>', 'There', 'are', 'three', 'sets', 'of', 'lights', 'per', 'direction', 'of', 'travel', '.', '</S>']
# ]

# orig_tgt = [
#     ['</S>', 'Pro', 'Fahr@@', 'tri@@', 'chtung', 'gibt', 'es', 'drei', 'Lich@@']
# ]

# orig_src = [
#     ['</S>', 'It', 'would', 'thus', 'be', 'suitable', 'to', 'assist', 'illegal', 'immigration', 'into', 'the', 'USA', '.', '</S>']
# ]

# orig_tgt = [
#     ['</S>', 'Er', 'wäre', 'damit', 'auch', 'geeignet', 'gewesen', ',', 'um', 'die', 'illegale', 'Einwanderung', 'Richtung', 'USA', 'zu', 'fördern', '.']
# ]

orig_src = [
    ['It', '&apos;s', 'important', 'to', 'use', 'the', 'term', 'business', '@-@', 'model', 'here', ':', 'Gam@@', 'ers', 'like', 'L@@', 'u', 'are', 'not', 'eng@@', 'ul@@', 'fed', 'by', 'the', 'Matrix', 'or', 'something', 'similar', '.', 'They', 'chose', 'to', 'play', 'a', 'game', 'with', 'a', 'specific', 'business', '@-@', 'model', ',', 'on', 'a', 'market', 'that', 'has', 'plenty', 'of', 'competition', 'and', 'choices', '-', 'both', 'Western', 'and', 'Asian', '.', 'Even', 'if', 'Southern', 'Week@@', 'end', ',', 'a', 'popular', 'tab@@', 'lo@@', 'id', 'in', 'China', 'puts', 'it', 'in', 'a', 'somewhat', 'xen@@', 'opho@@', 'bic', 'reaction', ':', '&quot;', '&apos;', 'Chinese', 'gam@@', 'ers', 'are', 'an', 'un@@', 'welcome', 'species', 'on', 'European', 'and', 'American', 'servers', ',', '&apos;', 'said', 'a', 'game', 'manager', 'who', 'once', 'worked', 'on', 'World', 'of', 'Warcraft', '.', 'For', 'those', 'European', 'and', 'American', 'gam@@', 'ers', ',', 'Chinese', 'players', 'are', 'like', 'f@@', 'ear@@', 'some', 'pag@@', 'ans', '.']
]

orig_tgt = [
    ['Der', 'Begriff', 'Geschäfts@@', 'modell', 'ist', 'von', 'Bedeutung', 'hier', ',', 'denn', 'L@@', 'u', 'sind', 'nicht', 'das', 'Opfer', 'Der', 'Matrix', 'oder', 'eines', 'üb@@', 'len', 'Kontroll@@', 'systems', '-', 'sie', 'haben', 'sich', 'auf', 'einem', 'freien', 'Markt', 'für', 'ein', 'bestimmtes', 'Spiel', 'und', 'sein', 'Geschäfts@@', 'modell', 'entschieden', ';', 'einem', 'Markt', 'mit', 'mas@@', 'sen@@', 'haft', 'Konkurrenz', 'und', 'zahlreichen', 'Alternativen', ',', 'sowohl', 'im', 'Westen', 'als', 'auch', 'im', 'Osten', '.', 'Auch', 'wenn', 'die', 'in', 'China', 'verbreitete', 'Zeitschrift', 'Southern', 'Week@@', 'end', 'auf', 'leicht', 'xen@@', 'opho@@', 'bische', 'Weise', 'po@@', 'stu@@', 'liert', ':', '&quot;', '&apos;', 'Chinese', 'gam@@', 'ers', 'are', 'an', 'un@@', 'welcome', 'species', 'on', 'European', 'and', 'American', 'servers', ',', '&apos;', 'said', 'a', 'game', 'manager', 'who', 'once', 'worked', 'on', 'World', 'of', 'Warcraft', '.', 'For', 'those', 'European', 'and', 'American', 'gam@@', 'ers', ',', 'Chinese', 'players', 'are', 'like', 'f@@', 'ear@@', 'some', 'pag@@', 'ans', '.']
]


src = [vocab_src.tokenize(s) for s in orig_src]
tgt = [vocab_tgt.tokenize(t) for t in orig_tgt]

print('tokenized src')
for s in src:
    print(s, len(s))

print('tokenized tgt')
for t in tgt:
    print(t, len(t))

src = torch.Tensor(src).to(torch.int32)
tgt = torch.Tensor(tgt).to(torch.int32)

model.eval()

with torch.no_grad():

    masks, out_mask = model.create_masks(src, tgt, pad_index)
    output, matrices = model.get_matrices(src, tgt, **masks)

    p_enc, p_src, p_tgt = calculate_patterns(model, matrices)
    A, B, C = summarize_matrices(model, matrices)

# plot(p_enc, p_src, p_tgt, orig_src[0], orig_tgt[0], dest='/home/fschmidt/patterns.png')
# plot(A, C, B, orig_src[0], orig_tgt[0], dest='/home/fschmidt/matrices.png')

encoder_self_matrices = [matrices[f'encoder/layer{l}/self_att']['value'] for l in range(config['encL'])]
decoder_self_matrices = [matrices[f'decoder/layer{l}/self_att']['value'] for l in range(config['decL'])]
decoder_cross_matrices = [matrices[f'decoder/layer{l}/cross_att']['value'] for l in range(config['decL'])]

plot_all_matrices(encoder_self_matrices, name='encoder_self_k16')
# plot_all_matrices(decoder_self_matrices, orig_src[0], orig_tgt[0], name='decoder_self')
# plot_all_matrices(decoder_cross_matrices, orig_src[0], orig_tgt[0], name='decoder_cross')

print('Done!')