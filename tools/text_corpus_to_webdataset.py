# python3 /home/fschmidt/code/deept/tools/text_corpus_to_webdataset.py \
# --corpus-files /home/fschmidt/data/wmt14/en-de/classical/train.en /home/fschmidt/data/wmt14/en-de/classical/train.de \
# --sample-names source target \
# --output-folder /home/fschmidt/data/wmt14/en-de/webdataset/train \
# --output-name train --number-of-shards 16

import _setup_env

import random
import argparse

import webdataset as wds
from os.path import isdir, isfile, join

from deept.util.debug import my_print

def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus-files', type=str, nargs='+', required=True,
        help="""The text corpus files that shall be combined into one webdataset.
        The text file entries are expected to be parallel so that the number of lines matches.
        The file name is used to identify the files in the webdataset.""")

    parser.add_argument('--sample-names', type=str, nargs='+', required=False, default=None,
        help="""The names that identify the samples from the individual files. 
        If no names are provided we will determine the identifiers based on the corpus file names.""")

    parser.add_argument('--output-folder', type=str, required=True,
        help="""The target directory of the webdataset.""")

    parser.add_argument('--output-name', type=str, default='webdataset', required=False,
        help="""The name of the created webdataset. Dont append '.tar'.""")

    parser.add_argument('--compress', type=bool, default=True, required=False,
        help="""Should the .tar content be compressed.""")

    parser.add_argument('--user', type=str, default='u', required=False,
        help="""User id of the user that created the .tar.""")

    parser.add_argument('--group', type=str, default='g', required=False,
        help="""Group id of the user that created the .tar.""")

    parser.add_argument('--number-of-shards', type=int, default=1, required=False,
        help="""If larger than one, it will shuffle the dataset and create 
            OUTPUT_NAME-shard{0...number_of_shards-1}.tar many files.
            It should only be done for train and dev sets and is needed 
            if you want to use multi-gpu training.""")

    parser.add_argument('--shuffle-before-sharding-buffer-size', type=int, default=1000000, required=False,
        help="""We will shuffle and buffer so many samples before they are divided into shards.""")

    args = parser.parse_args()

    return vars(args)

def start(args):
    
    check(args)
    write(args)

def check(args):

    for path in args['corpus_files']:
        assert isfile(path)
    
    assert isdir(args['output_folder'])

def write(args):
    
    files = [open(path, 'r') for path in args['corpus_files']]

    if args['sample_names'] is None:
        names = ['.'.join(path.split('/')[-1].split('.')[:-1]) for path in args['corpus_files']]
    else:
        names = args['sample_names']

    num_files = len(files)

    assert len(names) == num_files

    sink_kwargs = {
        'keep_meta': False,
        'compress': args['compress'],
        'user': args['user'],
        'group': args['group'],
        'mtime': None
    }

    nb_of_sinks = args['number_of_shards']

    if nb_of_sinks > 1:
        sinks = [wds.TarWriter(
                    join(args['output_folder'], f'{args["output_name"]}-{i:04d}.tar'), **sink_kwargs
                ) for i in range(nb_of_sinks)]
    else:
        sinks = [wds.TarWriter(join(args['output_folder'], args['output_name'] + '.tar'), **sink_kwargs)]

    buffer = []
    rng = random.Random()
    buffer_size = args['shuffle_before_sharding_buffer_size']
    write_i = 0

    assert len(sinks) == nb_of_sinks

    for i, lines in enumerate(zip(*files)):

        print(f'Read {i} samples.', end='\r')
        
        lines = list(lines)
        
        assert len(lines) == num_files

        sample = {
            "__key__": f"sample{i:010d}",
        }

        for j in range(num_files):
            sample[names[j]] = lines[j]

        if len(buffer) < buffer_size:
            buffer.append(sample)
        else:
            idx = rng.randint(0, len(buffer)-1)
            val, buffer[idx] = buffer[idx], sample
            sinks[write_i%nb_of_sinks].write(val)
            write_i += 1
    
    while buffer:
        idx = rng.randint(0, len(buffer)-1)
        sinks[write_i%nb_of_sinks].write(buffer.pop(idx))
        write_i += 1
    
    for sink in sinks:
        sink.close()

    for file in files:
        file.close()

if __name__ == '__main__':

    my_print(''.center(60, '-'))
    my_print(' Hi! '.center(60, '-'))
    my_print(' Script: text_corpus_to_webdataset.py '.center(60, '-'))
    my_print(''.center(60, '-'))

    args = parse_cli_arguments()

    start(args)

    my_print('Done!')
