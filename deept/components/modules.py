import math

import torch
import torch.nn as nn

from deept.utils.globals import Settings


class MultiHeadAttention(nn.Module):

    def __init__(self, H, D, dropout):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H
        
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, q, k, v, m=None):
        
        B = q.shape[0]
        D = self.D

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = q.view(B, -1, self.H, self.Dh)
        k = k.view(B, -1, self.H, self.Dh)
        v = v.view(B, -1, self.H, self.Dh)

        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)

        k = torch.transpose(k, -2, -1)

        a = torch.matmul(q, k)
        a = a / math.sqrt(D)

        if m is not None:
            a = a.masked_fill(m, -float('inf'))

        a = self.softmax(a)
        a = self.dropout(a)

        o = torch.matmul(a, v)

        o = torch.transpose(o, 1, 2)
        o = o.reshape(B, -1, self.D)
        o = self.W_o(o)

        return o, a


class GatedMultiHeadAttention(nn.Module):

    def __init__(self,
        H, D, dropout,
        gating=False,
    ):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H
        
        self.gating = gating

        self.__create_learnable_parameters(D, gating)
        self.__create_normalizations(D, gating)
        self.__create_activations(gating)

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def __create_learnable_parameters(self, D, gating):

        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)

        if gating:
            self.W_g = nn.Linear(D, D)
        else:
            self.W_g = nn.Identity()
    
    def __create_normalizations(self, D, gating):
        if gating:
            self.lnorm_v = LayerNormalization(self.D)
        else:
            self.lnorm_v = nn.Identity()

    def __create_activations(self, gating):
        if gating:
            self.act_v = nn.GELU()
            self.act_g = nn.GELU()
        else:
            self.act_v = nn.Identity()
            self.act_g = nn.Identity()

    def __call__(self, q, k, v, g, m=None):
        
        B = q.shape[0]
        H = self.H
        D = self.D
        Dh = self.Dh

        q = self.W_q(q)
        k = self.W_k(k)

        v = self.W_v(v)
        g = self.W_g(g)

        v = self.act_v(v)
        g = self.act_g(g)

        v = self.lnorm_v(v)

        q = q.view(B, -1, H, Dh)
        k = k.view(B, -1, H, Dh)
        v = v.view(B, -1, H, Dh)

        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)

        if self.gating:
            g = g.view(B, -1, H, Dh)
            g = torch.transpose(g, 1, 2)

        k = torch.transpose(k, -2, -1)

        a = torch.matmul(q, k)
        a = a / math.sqrt(Dh)

        if m is not None:
            a = a.masked_fill(m, -float('inf'))

        a = self.softmax(a)
        a = self.dropout(a)

        o = torch.matmul(a, v)

        if self.gating:
            o = o * g

        o = torch.transpose(o, 1, 2)
        o = o.reshape(B, -1, D)
        o = self.W_o(o)

        return o, a


class PositionalEmbedding(nn.Module):

    def __init__(self, V, model_dim, maxI, dropout, pad_index, use_pos_embed=True, return_pos_embed=False):
        super().__init__()

        self.model_dim = model_dim
        self.return_pos_embed = return_pos_embed
        self.use_pos_embed = use_pos_embed

        self.word_embed = nn.Embedding(V, model_dim, padding_idx=pad_index)
        if self.use_pos_embed or self.return_pos_embed:
            self.pos_embed = nn.Embedding(maxI, model_dim)
        self.dropout = nn.Dropout(dropout)

        rng = torch.arange(maxI)
        self.register_buffer('rng', rng)

    def __call__(self, x, J=None):

        B = x.shape[0]
        D = self.model_dim

        x = self.word_embed(x)

        if self.use_pos_embed or self.return_pos_embed:
            
            if J is None:
                J = x.shape[1]
                pos = self.pos_embed(self.rng[:J]) 
            else:
                assert x.shape[1] == 1
                pos = self.pos_embed(self.rng[J-1])

            pos = pos.unsqueeze(0).repeat(B, 1, 1)

            if self.use_pos_embed:
                x = x + pos
        
        x = x * math.sqrt(D)

        x = self.dropout(x)

        if self.return_pos_embed:
            return x, pos
        else:
            return x


class SinusodialPositionalEmbedding(nn.Module):

    def __init__(self, V, D, maxI, dropout, pad_index, use_pos_embed=True, return_pos_embed=False):
        super().__init__()

        self.D = D
        self.maxI = maxI

        self.return_pos_embed = return_pos_embed
        self.use_pos_embed = use_pos_embed

        self.word_embed = nn.Embedding(V, D, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout)

        self.__precalculate_pos_embed(D, maxI)

    def __precalculate_pos_embed(self, D, maxI):

        import math

        pos_embed = torch.arange(maxI, dtype=torch.float).unsqueeze(-1).repeat(1, D)
        pos = torch.arange(maxI, dtype=torch.float).unsqueeze(-1)
        i = torch.arange(0, D, 2, dtype=torch.float)

        i = torch.exp(-(i / D) * math.log(10000))
        pos = pos * i

        pos_embed[:, 0::2] = torch.sin(pos)
        pos_embed[:, 1::2] = torch.cos(pos)

        self.register_buffer('pos_embed', pos_embed.to(Settings.get_device()))

    def __call__(self, x, J=None):

        B = x.shape[0]
        D = self.D

        x = self.word_embed(x)

        x = x * math.sqrt(D)

        if self.use_pos_embed or self.return_pos_embed:

            if J is None:
                J = x.shape[1]
                assert J <= self.maxI, f"""Error! x exceeded maximum sequence length! x.shape:{x.shape}"""
                pos = self.pos_embed[:J]
            else:
                assert x.shape[1] == 1
                pos = self.pos_embed[J-1]

            pos = pos.unsqueeze(0).repeat(B, 1, 1)  

            if self.use_pos_embed:
                x = x + pos

        x = self.dropout(x)

        if self.return_pos_embed:
            return x, pos
        else:
            return x


class LayerNormalization(nn.Module):

    def __init__(self, model_dim):
        super().__init__()

        self.a = nn.Parameter(torch.ones(model_dim))
        self.b = nn.Parameter(torch.zeros(model_dim))

    def __call__(self, x):
        
        mu = torch.mean(x, dim=-1, keepdim=True)
        sg = torch.var(x, dim=-1, keepdim=True)

        x = (x - mu) / torch.sqrt(sg + 1e-8)
        x = x * self.a + self.b

        return x