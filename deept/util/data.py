

import pickle

import torch
import torch.distributed as dist
import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterDataPipe

from deept.util.debug import my_print
from deept.util.globals import Settings, Context


__DP_DECODING__ = {}
__DP_PREPROCESSING__ = {}
__DP_OVERWRITE__ = {}
__LEN_FN__ = {}


def register_dp_decoding(name):
    def register_dp_decoding_fn(cls):
        if name in __DP_DECODING__:
            raise ValueError(f'Decoding datapipe {name} already registered!')
        __DP_DECODING__[name] = cls
        return cls
    return register_dp_decoding_fn

def register_dp_preprocessing(name):
    def register_dp_preprocessing_fn(cls):
        if name in __DP_PREPROCESSING__:
            raise ValueError(f'Preprocessing datapipe {name} already registered!')
        __DP_PREPROCESSING__[name] = cls
        return cls
    return register_dp_preprocessing_fn

def register_dp_collate(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn

def register_len_fn(name):
    def register_len_fn_fn(cls):
        if name in __LEN_FN__:
            raise ValueError(f'Len function {name} already registered!')
        __LEN_FN__[name] = cls
    return register_len_fn_fn

def register_dp_overwrite(name):
    def register_dp_overwrite_fn(cls):
        if name in __DP_OVERWRITE__:
            raise ValueError(f'Overwrite datapipe {name} already registered!')
        __DP_OVERWRITE__[name] = cls
        return cls
    return register_dp_overwrite_fn


def create_dp_from_config(config, data_root, data_mask, bucket_batch=False):

    user_dp_overwrite_key = config['data_dp_overwrite', '']
    if user_dp_overwrite_key != '' and user_dp_overwrite_key in __DP_OVERWRITE__:
        return create_dp_overwrite_from_config(config)

    pipe = (
        dp.iter.FileLister(root=data_root, masks=data_mask, recursive=False, abspath=True)
        .shuffle(buffer_size=10000) # shuffle shards
        .open_files(mode="b")
        .load_from_tar()
    )

    pipe = create_decoding_dp_from_config(config, pipe)

    pipe = (
        pipe.webdataset()
        .shuffle(buffer_size=10000) # shuffle shards
        .sharding_filter() # Distributes across processes
    )

    pipe = create_preprocessing_dp_from_config(config, pipe)
    len_fn = get_len_fn(config)

    pipe = (
        pipe.max_token_bucketize(max_token_count=config['batch_size'], len_fn=len_fn, include_padding=False)
        .shuffle(buffer_size=30)
    )

    # pipe = create_collating_dp_from_config(config)

    return pipe

def create_decoding_dp_from_config(config, source_dp):
    if config['data_decoding'] in __DP_DECODING__:
        pipe = __DP_DECODING__[config['data_decoding']].create_from_config(config, source_dp)
        return pipe
    else:
        raise ValueError(f'Error! Unrecognized decoding datapipe {config["data_decoding"]}!')

def create_preprocessing_dp_from_config(config, source_dp):
    if config['data_preprocess'] in __DP_PREPROCESSING__:
        pipe = __DP_PREPROCESSING__[config['data_preprocess']].create_from_config(config, source_dp)
        return pipe
    else:
        raise ValueError(f'Error! Unrecognized preprocessing datapipe {config["data_preprocess"]}!')

def create_collating_dp_from_config(config):
    pass

def create_dp_overwrite_from_config(config):
    user_dp_overwrite_key = config['data_dp_overwrite']
    datapipe = __DP_OVERWRITE__[user_dp_overwrite_key].create_from_config(config)
    return datapipe

def get_len_fn(config):
    if config['data_len_fn'] in __LEN_FN__:
        return __LEN_FN__[config['data_len_fn']]
    else:
        raise ValueError(f'Error! Unrecognized length function {config["data_len_fn"]}!')


def get_all_dp_decoding_keys():
    return list(__DP_DECODING__.keys())

def get_all_dp_preprocessing_keys():
    return list(__DP_PREPROCESSING__.keys())

def get_all_len_fn_keys():
    return list(__LEN_FN__.keys())

def get_all_dp_overwrite_keys():
    return list(__DP_OVERWRITE__.keys())


@register_len_fn('mt_tgt_len_fn')
def mt_length_function(item):
    return len(item['tgt'])


@register_dp_decoding('text')
class WebDatasetTextDecoderIterDataPipe(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    @staticmethod
    def create_from_config(config, source_dp):
        return WebDatasetTextDecoderIterDataPipe(source_dp)

    def __iter__(self):
        for d in self.source_dp:
            yield self.webdataset_text_decode(d)

    def webdataset_text_decode(self, item):
        key, value = item
        return key, value.read().decode('utf-8')

    def __len__(self):
        raise NotImplementedError('Error! Do not invoke len(datapipe).')


class MTVocabulary:

    UNK = "<UNK>"
    PAD = "<PAD>"
    EOS = "</S>"

    special_tokens = [
        UNK,
        PAD,
        EOS
    ]

    def __init__(self, vocab, vocab_size, vocab_rev, vocab_path):
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.vocab_rev = vocab_rev
        self.vocab_path = vocab_path

        self.PAD = self.vocab[MTVocabulary.PAD]
        self.UNK = self.vocab[MTVocabulary.UNK]
        self.EOS = self.vocab[MTVocabulary.EOS]

    @staticmethod
    def create_vocab(vocab_path):

        vocab = MTVocabulary.read_from_pickle(vocab_path)
        vocab = MTVocabulary.append_special_tokens(vocab)
        vocab = MTVocabulary.remove_sos_symbol_from_vocabs(vocab)

        vocab_size = len(vocab.items())
        vocab_rev = {y:x for x,y in vocab.items()}

        return MTVocabulary(vocab, vocab_size, vocab_rev, vocab_path)

    @staticmethod
    def read_from_pickle(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    @staticmethod
    def append_special_tokens(vocab):

        count = 0
        for special_token in MTVocabulary.special_tokens:
            if special_token not in vocab.keys():
                count += 1

        MTVocabulary.increment_dictionary(vocab, count)

        new_index = 0
        for special_token in MTVocabulary.special_tokens:
            if special_token not in vocab.keys():
                my_print(f'Inserting special token {special_token} to vocabulary at position {new_index}.')
                vocab[special_token] = new_index
                new_index += 1

        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

        return vocab

    @staticmethod
    def increment_dictionary(dictionary, increment):

        assert isinstance(dictionary, dict)
        assert isinstance(increment, int)

        for entry in dictionary:

            assert isinstance(dictionary[entry], int)

            dictionary[entry] += increment
    
    @staticmethod
    def remove_sos_symbol_from_vocabs(vocab):
        if "<S>" in vocab:
            del vocab["<S>"]
        return vocab

    def print_vocab(self):
        for k,v in self.vocab.items():
            my_print(k,v)

    def tokenize(self, inp):
        if isinstance(inp, list):
            return self.tokenize_list(inp)
        elif isinstance(inp, str):
            return self.tokenize_word(inp)
        else:
            assert True == False, 'Got unknown input for tokenization.'

    def tokenize_list(self, line):

        assert isinstance(line, list)

        tokenized = []
        for word in line:
            tokenized.append(self.tokenize_word(word))

        return tokenized

    def tokenize_word(self, word):
        if word in self.vocab:
            return int(self.vocab[word])
        else:
            return int(self.vocab[MTVocabulary.UNK])


    def detokenize(self, inp):
        if isinstance(inp, list):
            return self.detokenize_list(inp)
        elif isinstance(inp, str):
            return self.detokenize_word(inp)
        else:
            assert True == False, 'Got unknown input for detokenization.'

    def detokenize_list(self, line):

        assert isinstance(line, list)

        detokenized = []
        for word in line:
            detokenized.append(self.detokenize_word(word))

        return detokenized

    def detokenize_word(self, word):
        return self.vocab_rev[word]

    
    def remove_padding(self, sentence):
        ret = []
        for word in sentence:
            if word != MTVocabulary.PAD:
                ret.append(word)
        return ret


@register_dp_preprocessing('mt_preprocess')
class MTPreprocesserIterDataPipe(IterDataPipe):

    def __init__(self, source_dp, vocab_src, vocab_tgt):
        super().__init__()
        self.source_dp = source_dp
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    @staticmethod
    def create_from_config(config, source_dp):

        from deept.util.globals import Context

        vocab_src = MTVocabulary.create_vocab(config['vocab_src'])
        vocab_tgt = MTVocabulary.create_vocab(config['vocab_tgt'])

        Context.add_context('vocab_src', vocab_src)
        Context.add_context('vocab_tgt', vocab_tgt)

        my_print(f'Vocab size source {vocab_src.vocab_size}!')
        my_print(f'Vocab size target {vocab_tgt.vocab_size}!')

        return MTPreprocesserIterDataPipe(
            source_dp,
            vocab_src,
            vocab_tgt
        )

    def __iter__(self):
        for item in self.source_dp:
            yield self.mt_preprocess(self.normalize_keys(item))

    def normalize_keys(self, item):
        
        assert isinstance(item, dict), """The webdataset format presets that every sample
            is a dictionary."""

        assert '__key__' in item.keys(), """The webdataset format presets that every sample
            is a dictionary and contains a __key__ entry."""

        idx = item['__key__'].split('/')[-1]

        assert 'sample' in idx, """At the moment we expect you to name each sample of the webdataset sampleXXXXX."""

        idx = int(idx.split('/')[-1].split('sample')[-1])

        normalized_dict = {
            '__key__': idx
        }

        for k, v in item.items():
            if 'source' in k:
                normalized_dict['src'] = v
            elif 'target' in k:
                normalized_dict['tgt'] = v

        return normalized_dict

    def mt_preprocess(self, item):

        item['src'] = item['src'].strip().replace("\n", "").split(" ")
        item['tgt'] = item['tgt'].strip().replace("\n", "").split(" ")

        item['src'] = self.vocab_src.tokenize(item['src'])
        item['tgt'] = self.vocab_src.tokenize(item['tgt'])

        return item

    def __len__(self):
        raise NotImplementedError('Error! Do not invoke len(datapipe).')


@register_dp_preprocessing('mt_collate')
class MTCollaterIterDataPipe(IterDataPipe):

    def __init__(self, source_dp, vocab_src, vocab_tgt):
        super().__init__()
        self.source_dp = source_dp
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    @staticmethod
    def create_from_config(config, source_dp):

        from deept.util.globals import Context

        vocab_src = MTVocabulary.create_vocab(config['vocab_src'])
        vocab_tgt = MTVocabulary.create_vocab(config['vocab_tgt'])

        Context.add_context('vocab_src', vocab_src)
        Context.add_context('vocab_tgt', vocab_tgt)

        my_print(f'Vocab size source {vocab_src.vocab_size}!')
        my_print(f'Vocab size target {vocab_tgt.vocab_size}!')

        return MTPreprocesserIterDataPipe(
            source_dp,
            vocab_src,
            vocab_tgt
        )

    def __iter__(self):
        for item in self.source_dp:
            yield self.mt_preprocess(self.normalize_keys(item))

    def normalize_keys(self, item):
        
        assert isinstance(item, dict), """The webdataset format presets that every sample
            is a dictionary."""

        assert '__key__' in item.keys(), """The webdataset format presets that every sample
            is a dictionary and contains a __key__ entry."""

        idx = item['__key__'].split('/')[-1]

        assert 'sample' in idx, """At the moment we expect you to name each sample of the webdataset sampleXXXXX."""

        idx = int(idx.split('/')[-1].split('sample')[-1])

        normalized_dict = {
            '__key__': idx
        }

        for k, v in item.items():
            if 'source' in k:
                normalized_dict['src'] = v
            elif 'target' in k:
                normalized_dict['tgt'] = v

        return normalized_dict

    def mt_preprocess(self, item):

        item['src'] = item['src'].strip().replace("\n", "").split(" ")
        item['tgt'] = item['tgt'].strip().replace("\n", "").split(" ")

        item['src'] = self.vocab_src.tokenize(item['src'])
        item['tgt'] = self.vocab_src.tokenize(item['tgt'])

        return item

    def __len__(self):
        raise NotImplementedError('Error! Do not invoke len(datapipe).')
