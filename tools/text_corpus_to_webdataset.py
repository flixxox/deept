# python3 /home/fschmidt/code/deept/tools/text_corpus_to_webdataset.py \
# --corpus-files /home/fschmidt/data/iwslt/de-en/train.de /home/fschmidt/data/iwslt/de-en/train.en \
# --sample-names source target \
# --output-folder /home/fschmidt/data/iwslt/de-en/webdataset --output-name train.tar

import _setup_env

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
    parser.add_argument('--output-name', type=str, default='webdataset.tar', required=False,
        help="""The name of the created webdataset.""")
    parser.add_argument('--compress', type=bool, default=True, required=False,
        help="""Should the .tar content be compressed.""")
    parser.add_argument('--user', type=str, default='u', required=False,
        help="""User id of the user that created the .tar.""")
    parser.add_argument('--group', type=str, default='g', required=False,
        help="""Group id of the user that created the .tar.""")

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

    sink = wds.TarWriter(join(args['output_folder'], args['output_name']),
        keep_meta=False,
        compress=args['compress'],
        user=args['user'],
        group=args['group'],
        mtime=None
    )

    for i, lines in enumerate(zip(*files)):
        
        lines = list(lines)
        
        assert len(lines) == num_files

        to_write = {
            "__key__": f"sample{i:10d}",
        }

        for j in range(num_files):
            to_write[names[j]] = lines[j]

        sink.write(to_write)

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
