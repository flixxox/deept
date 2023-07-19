
import torch
import torch.distributed as dist

from deept.util.debug import my_print
from deept.util.globals import Settings, Context


def register_dp_decoding(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn

def register_dp_preprocessing(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn

def register_dp_collate(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn

def register_dp_len_fn(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

def register_dp_overwrite(name):
    def register_model_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f'Model {name} already registered!')
        __MODEL_DICT__[name] = cls
        return cls

    return register_model_fn


def create_dp_from_config(config, train=False):
    
    if do_dp_overwrite():
        return create_dp_overwrite_from_config(config)

    decoding_fn = create_decoding_dp_from_config(config)
    preprocessing_fn = create_preprocessing_dp_from_config(config)
    collate_fn = create_collating_dp_from_config(config)
    len_fn = LEN_FN[config['len_fn']]

    pipe = (
        dp.iter.FileLister(root='/home/fschmidt/data/iwslt/de-en/webdataset', masks='train.tar', recursive=False, abspath=True)
        .shuffle(buffer_size=10000) # shuffle shards
        .open_files(mode="b")
        .load_from_tar()
        .map(decoding_fn)
        .webdataset()
        .shuffle(buffer_size=10000) # shuffle shards
        .sharding_filter() # Distributes across processes
        .map(preprocessing_fn)
        .max_token_bucketize(max_token_count=100, len_fn=len_fn, include_padding=False)
        .shuffle(buffer_size=30)
        .map(collate_fn) # use the collate functionality for this
    )

    return pipe

def create_decoding_dp_from_config(config):
    pass

def create_preprocessing_dp_from_config(config):
    pass

def create_collating_dp_from_config(config):
    pass

def create_dp_overwrite_from_config(config):
    pass


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

        self.PAD = self.vocab[Vocabulary.PAD]
        self.UNK = self.vocab[Vocabulary.UNK]
        self.EOS = self.vocab[Vocabulary.EOS]

    @staticmethod
    def create_vocab(vocab_path):

        vocab = Vocabulary.read_from_pickle(vocab_path)
        vocab = Vocabulary.append_special_tokens(vocab)
        vocab = Vocabulary.remove_sos_symbol_from_vocabs(vocab)

        vocab_size  = len(vocab.items())
        vocab_rev   = {y:x for x,y in vocab.items()}

        return Vocabulary(vocab, vocab_size, vocab_rev, vocab_path)

    @staticmethod
    def read_from_pickle(vocab_path):

        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    @staticmethod
    def append_special_tokens(vocab):

        count = 0
        for special_token in Vocabulary.special_tokens:
            if special_token not in vocab.keys():
                count += 1

        Vocabulary.increment_dictionary(vocab, count)

        new_index = 0
        for special_token in Vocabulary.special_tokens:
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
            return int(self.vocab[Vocabulary.UNK])


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
            if word != Vocabulary.PAD:
                ret.append(word)
        return ret

