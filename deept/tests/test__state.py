import unittest

import torch
import numpy as np

from deept.util.globals import Globals
from deept.model.state import DynamicState

class TestState(unittest.TestCase):


    def test_full(self):

        Globals.set_train_flag(False)

        state = DynamicState(time_dim=1)

        B = 2
        N = 3
        D1 = 4
        D2 = 5

        steps = 3

        for i in range(steps):

            s = torch.arange(B*N*D1*D2, dtype=torch.float32) * (i+1)
            s = s.view(-1, 1, D1, D2)

            s = state.full(s)

            assert list(s.shape) == [B*N, i+1, D1, D2]

    def test_reorder(self):

        import math
        
        Globals.set_train_flag(False)

        state = DynamicState()

        B = 2
        N = 3
        D = [4, 5, 6]

        reorder = torch.Tensor([
            5,2,1,0,4,5
        ]).to(torch.int64)

        assert list(reorder.shape) == [B*N]

        for i in range(1, len(D)+1):
            Dc = D[:i]

            prod = math.prod(Dc)

            state.cache = torch.arange(B*N*prod, dtype=torch.float32)
            state.cache = state.cache.view(-1, *Dc)

            assert list(state.cache.shape) == [B*N, *Dc]

            exp_cache = []
            for i in reorder.numpy():
                exp_cache.append(list(state.cache[i].numpy()))
            exp_cache = np.array(exp_cache)

            assert list(exp_cache.shape) == [B*N, *Dc]

            state.reorder(reorder)

            np.testing.assert_array_equal(state.cache, exp_cache)

    def test_reduce__no_cummulate(self):

        import math

        Globals.set_train_flag(False)

        state = DynamicState()

        B = 2
        N = 3

        D = [4, 5, 6]

        mask = torch.Tensor([
            True, False, True, False, True, False
        ]).to(torch.bool)

        assert list(mask.shape) == [B*N]

        for i in range(1, len(D)+1):
            Dc = D[:i]

            prod = math.prod(Dc)

            state.cache = torch.arange(B*N*prod, dtype=torch.float32)
            state.cache = state.cache.view(-1, *Dc)

            assert list(state.cache.shape) == [B*N, *Dc]

            exp_cache = []
            for i in range(mask.shape[0]):
                if mask[i]:
                    exp_cache.append(state.cache[i].numpy())
            exp_cache = np.array(exp_cache)

            assert list(exp_cache.shape) == [3, *Dc]

            state.reduce(mask, cummulate_masks=False)

            np.testing.assert_array_equal(state.cache, exp_cache)

    def test_reduce__cummulate(self):

        import math

        Globals.set_train_flag(False)

        state = DynamicState()

        B = 2
        N = 3

        D = [4, 5, 6]

        prev_mask = torch.Tensor([
            True, True, True, False, True, False, True, False, True, False, False
        ]).to(torch.bool)

        mask = torch.Tensor([
            False, False, True, True, False, False, True, True, False, False, True
        ]).to(torch.bool)

        for i in range(1, len(D)+1):
            Dc = D[:i]

            prod = math.prod(Dc)

            state.cache = torch.arange(B*N*prod, dtype=torch.float32)
            state.cache = state.cache.view(-1, *Dc)

            assert list(state.cache.shape) == [B*N, *Dc]

            exp_cache = []
            i = 0
            for m_prev, m  in zip(prev_mask, mask):
                if m_prev:
                    if m:
                        exp_cache.append(state.cache[i].numpy())
                    i += 1
            exp_cache = np.array(exp_cache)

            assert list(exp_cache.shape) == [2, *Dc]
            
            state.prev_mask = prev_mask
            state.reduce(mask)

            np.testing.assert_array_equal(state.cache, exp_cache)