import unittest

import numpy as np
import torch

from deept.util.config import Config
from deept.util.debug import my_print
from deept.util.data import Vocabulary
from deept.model.state import DynamicState
from deept.search.beam_search import BeamSearch

class TestBeamSearch(unittest.TestCase):

    def test_get_active_model_input(self):
        
        B = 3
        srcT = 6
        i = 5
        N = 4
        V = 10
        D = 2
        maxI = 3
        EOS = 7
        PAD = -1
        nencs = 3

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        src = torch.arange(B*N*srcT)
        src = src.view(B, N, srcT)

        tgt = torch.arange(B*N*i)
        tgt = tgt.reshape(B, N, i)

        active_mask = torch.tensor([
            [
                True, False, True, False
            ],
            [
                False, False, False, False
            ],
            [
                False, True, False, True
            ],
        ])
        BNa = 4

        assert list(src.shape) == [B, N, srcT]
        assert list(tgt.shape) == [B, N, i]
        assert list(active_mask.shape) == [B, N]

        exp_srca = torch.tensor([
            [ 0,  1,  2,  3,  4,  5],
            [12, 13, 14, 15, 16, 17],
            [54, 55, 56, 57, 58, 59],
            [66, 67, 68, 69, 70, 71],
        ])

        exp_tgta = torch.tensor([
            [ 0,  1,  2,  3,  4],
            [10, 11, 12, 13, 14],
            [45, 46, 47, 48, 49],
            [55, 56, 57, 58, 59],
        ])

        assert list(exp_srca.shape) == [BNa, srcT]
        assert list(exp_tgta.shape) == [BNa, i]

        srca, tgta = beam_search.get_active_model_input(src, tgt, active_mask)

        srca = srca.numpy()
        tgta = tgta.numpy()

        np.testing.assert_array_equal(srca, exp_srca)
        np.testing.assert_array_equal(tgta, exp_tgta)

    def test_pad_to_N(self):
        
        B = 3
        maxI = 3
        BNa = 4
        N = 4
        V = 4
        EOS = 7

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, -1)

        precomp_indices = torch.arange(N).unsqueeze(0)
        precomp_indices = precomp_indices.repeat(B, 1) + torch.arange(B).unsqueeze(1) * N

        active_mask = torch.Tensor([
            [True, False, True, False],
            [False, True, False, True],
            [False, False, False, False],
        ]).to(torch.bool)

        output = torch.Tensor([
                [ 1, 2, 3, 4],
                [ 5, 6, 7, 8],
                [13,14,15,16],
                [17,18,19,20],
            ])

        assert list(output.shape) == [BNa, V]
        assert list(active_mask.shape) == [B, N]
        assert list(precomp_indices.shape) == [B, N]

        exp_output = torch.Tensor([
            [
                [ 1, 2, 3, 4],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ 5, 6, 7, 8],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
            ],
            [
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [13,14,15,16],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [17,18,19,20],
            ],
            [
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
                [ -float('inf'), -float('inf'), -float('inf'), -float('inf')],
            ],
        ])

        assert list(exp_output.shape) == [B, N, V]

        output = beam_search.pad_to_N(output, active_mask, precomp_indices)

        output = output.numpy()
        exp_output = exp_output.numpy()

        np.testing.assert_array_equal(output, exp_output)
    
    def test_update_tgt(self):
        
        B = 3
        i = 3
        N = 3
        maxI = 7
        V = 10
        EOS = 9
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        tgt = torch.Tensor([
            [
                [1, 2,   9],
                [4, 5,   6],
                [7, 9, PAD],
            ],
            [
                [10, 11,  12],
                [13, 9,  PAD],
                [16, 17,  18],
            ],
            [
                [18, 19,   9],
                [20, 21,   9],
                [22,  9, PAD],
            ]
        ]).to(torch.int32)

        active_mask = torch.Tensor([
            [False, True, False],
            [True, False, True],
            [False, False, False],
        ]).to(torch.bool)

        best_beams = torch.Tensor([
            [1, 1, 1],
            [2, 0, 0],
            [0, 0, 0],
        ]).to(dtype=torch.int32)

        best_words = torch.Tensor([
            [  9, 100, 101],
            [102,   9, 103],
            [  0,   1,   2],
        ]).to(torch.int32)

        scores = torch.Tensor([
            [-1, -2, -3],
            [-4, -5, -6],
            [-7, -8, -9],
        ]).to(torch.float32)

        assert list(tgt.shape) == [B, N, i]
        assert list(active_mask.shape) == [B, N]
        assert list(best_beams.shape) == [B, N]
        assert list(best_words.shape) == [B, N]
        assert list(scores.shape) == [B, N]

        exp_tgt = torch.Tensor([
            [
                [1, 2,   9, PAD],
                [4, 5,   6,   9],
                [7, 9, PAD, PAD],
            ],
            [
                [16, 17,  18, 102],
                [13, 9,  PAD, PAD],
                [10, 11,  12,   9],
            ],
            [       
                [18, 19,   9, PAD],
                [20, 21,   9, PAD],
                [22,  9, PAD, PAD],
            ]
        ])

        exp_scores = torch.Tensor([
            [-float('inf'),            -1, -float('inf')],
            [           -4, -float('inf'),            -5],
            [-float('inf'), -float('inf'), -float('inf')],
        ])

        exp_best_words = torch.Tensor([
            [PAD,   9, PAD],
            [102, PAD,   9],
            [PAD, PAD, PAD],
        ])

        assert list(exp_tgt.shape) == [B, N, i+1]
        assert list(exp_scores.shape) == [B, N]
        assert list(exp_best_words.shape) == [B, N]

        tgt, scores, best_words = beam_search.update_tgt(tgt, active_mask, best_beams, best_words, scores)

        tgt = tgt.numpy()
        scores = scores.numpy()

        exp_tgt = exp_tgt.numpy()
        exp_scores = exp_scores.numpy()
        exp_best_words = exp_best_words.numpy()

        np.testing.assert_allclose(tgt, exp_tgt)
        np.testing.assert_allclose(scores, exp_scores)
        np.testing.assert_allclose(best_words, exp_best_words)

    def test_reorder_states(self):

        B = 3
        N = 4
        BNa = 9
        i = 2
        D = 3
        maxI = 7
        V = 10
        EOS = 9
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD, stepwise=True)

        beam_search.dynamic_states = [DynamicState()]

        beam_search.dynamic_states[0].cache = torch.arange(BNa*i*D, dtype=torch.float32).view(BNa, i, D)

        active_mask = torch.Tensor([
            [True, False, True, True],
            [True, False, True, False],
            [True, True, True, True]
        ]).to(torch.bool)

        best_beams = torch.Tensor([
            [0, 2, 3, 0],
            [2, 0, 2, 0],
            [3, 2, 1, 0],
        ]).to(torch.int32)

        exp_cache = torch.Tensor([
            [
                [ 0.,  1.,  2.],
                [ 3.,  4.,  5.]
            ],
            [
                [ 6.,  7.,  8.],
                [ 9., 10., 11.]
            ],
            [
                [12., 13., 14.],
                [15., 16., 17.]
            ],
            [
                [24., 25., 26.],
                [27., 28., 29.]
            ],
            [
                [18., 19., 20.],
                [21., 22., 23.]
            ],
            [
                [48., 49., 50.],
                [51., 52., 53.]
            ],
            [
                [42., 43., 44.],
                [45., 46., 47.]
            ],
            [
                [36., 37., 38.],
                [39., 40., 41.]
            ],
            [
                [30., 31., 32.],
                [33., 34., 35.]
            ]
        ]).to(torch.float32)

        assert list(beam_search.dynamic_states[0].cache.shape) == [BNa, i, D]
        assert list(exp_cache.shape) == [BNa, i, D]
        assert list(active_mask.shape) == [B, N]
        assert list(best_beams.shape) == [B, N]

        beam_search.reorder_states(best_beams, active_mask)

        cache = beam_search.dynamic_states[0].cache.numpy()

        np.testing.assert_allclose(cache, exp_cache)

    def test_update_fin(self):
        
        B = 4
        N = 3
        V = 10
        maxI = 8
        EOS = 9
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        fin_storage_scores = torch.Tensor([
            [-float('inf'),            -1, -float('inf')],
            [           -7, -float('inf'),            -2],
            [-float('inf'), -float('inf'), -float('inf')],
            [           -5,            -3,            -6],
        ]).to(torch.float32)

        active_mask = torch.Tensor([
            [True, False, True],
            [False, True, False],
            [True, True, True],
            [False, False, False],
        ]).to(torch.bool)

        best_words = torch.Tensor([
            [  1, PAD,   9],
            [PAD,   2, PAD],
            [  3,   9,   4],
            [PAD, PAD, PAD],
        ]).to(torch.int32)

        scores = torch.Tensor([
            [           10, -float('inf'),            14],
            [-float('inf'),            12, -float('inf')],
            [           11,            13,            15],
            [-float('inf'), -float('inf'), -float('inf')],
        ]).to(torch.float32)

        assert list(fin_storage_scores.shape) == [B, N]
        assert list(active_mask.shape) == [B, N]
        assert list(best_words.shape) == [B, N]
        assert list(scores.shape) == [B, N]

        exp_fin_storage_scores = torch.Tensor([
            [-float('inf'),            -1,            14],
            [           -7, -float('inf'),            -2],
            [-float('inf'),            13, -float('inf')],
            [           -5,            -3,            -6],
        ]).to(torch.float32)

        exp_active_mask = torch.Tensor([
            [True, False, False],
            [False, True, False],
            [True, False, True],
            [False, False, False],
        ]).to(torch.bool)

        assert list(exp_fin_storage_scores.shape) == [B, N]
        assert list(exp_active_mask.shape) == [B, N]

        fin_storage_scores, active_mask = beam_search.update_fin(fin_storage_scores, best_words, active_mask, scores)

        exp_fin_storage_scores = exp_fin_storage_scores.numpy()
        exp_active_mask = exp_active_mask.numpy()

        np.testing.assert_array_equal(fin_storage_scores, exp_fin_storage_scores)
        np.testing.assert_array_equal(active_mask, exp_active_mask)
    
    def test_select_best_hyp(self):
        
        B = 3
        N = 4
        i = 5
        V = 10
        maxI = 8
        EOS = 9
        PAD = -1

        beam_search = self.__setup_test_object(B, maxI, N, V, EOS, PAD)

        tgt = torch.arange(B*N*i)
        tgt = tgt.view(B, N, i)

        fin_storage_scores = torch.Tensor([
            [
                -4,
                -3,
                 0,
                -1
            ],
            [
                -1000,
                -float('inf'),
                -float('inf'),
                -float('inf'),
            ],
            [
                -7,
                -5,
                -6,
                -5.1
            ],
        ]).to(torch.float32)

        assert list(tgt.shape) == [B, N, i]
        assert list(fin_storage_scores.shape) == [B, N]

        exp_tgt = torch.Tensor([
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [45, 46, 47, 48, 49],
        ]).to(torch.int32)

        tgt = beam_search.select_best_hyp(tgt, fin_storage_scores)

        tgt = tgt.numpy()
        exp_tgt = exp_tgt.numpy()

        np.testing.assert_array_equal(tgt, exp_tgt)

    def __setup_test_object(self, B, maxI, N, V, EOS, PAD, stepwise=False):

        hvd.init()

        path_test_vocab_src = 'tests/res/mockup_vocab_src.pickle'
        path_test_vocab_tgt = 'tests/res/mockup_vocab_tgt.pickle'

        config = Config(
            {
                'batch_size_search': B,
                'max_sentence_length': maxI,
                'beam_size': N,
                'length_norm': True,
                'model_dim': 20,
                'stepwise': stepwise
            }
        )

        vocab_src = Vocabulary.create_vocab(path_test_vocab_src)
        vocab_tgt = Vocabulary.create_vocab(path_test_vocab_tgt)

        vocab_tgt.vocab_src = V
        vocab_tgt.vocab_size = V

        vocab_src.EOS = EOS
        vocab_tgt.EOS = EOS

        vocab_src.PAD = PAD
        vocab_tgt.PAD = PAD

        beam_search = BeamSearch.create_search_algorithm_from_config(config, None, vocab_src, vocab_tgt)

        return beam_search