import torch
from hrr import *
# from qwrapper.obs import PauliObservable
from torch import nn
from torch.nn.init import xavier_uniform_

def transpose_for_scores(x: torch.Tensor, nh, hhs) -> torch.Tensor:
    attention_head_size = hhs//nh
    new_x_shape = x.size()[:-1] + (nh, attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


class PauliEnergy(nn.Module):
    def __init__(
        self,
        hidden_size=9,
        attention_hidden_size=8,
        num_heads=4,
        num_gates=8000,
        num_paulis=128,
    ):
        super().__init__()
        self.hs = hidden_size
        self.hhs = attention_hidden_size
        self.nh = num_heads
        self.ml = num_gates
        self.bs = num_paulis
        # self.gate_embeddings = nn.Embedding(self.bs, self.hs) # original embedding is one hot
        self.positional_bias = nn.Parameter(torch.empty(1, self.ml, self.hs))
        xavier_uniform_(self.positional_bias)
        self.query = nn.Linear(self.hs, self.hhs)
        self.key = nn.Linear(self.hs, self.hhs)
        self.value = nn.Linear(self.hs, self.hhs)
        
        self.layernorm = nn.LayerNorm(self.hs, eps=1e-6)
        self.ffn1 = nn.Linear(self.hhs, self.hs)
        self.ffn2 = nn.Linear(self.hhs, self.hs)
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(self.hs, 1)

    def forward(self, x, mask=None):
        x = x + self.positional_bias
        x = self.layernorm(x)
        # k, q, v = torch.randn(bs, nh, ml, hs//nh), torch.randn(bs, nh, ml, hs//nh), torch.randn(bs, nh, ml, hs//nh)
        q = transpose_for_scores(self.query(x), self.nh, self.hhs)
        k = transpose_for_scores(self.key(x), self.nh, self.hhs)
        v = transpose_for_scores(self.value(x), self.nh, self.hhs)

        bind = binding(k, v, dim=-1).sum(dim=-2, keepdims=True)  # (B, h, 1, H')
        vp = unbinding(bind, q, dim=-1)  # (B, h, T, H')
        scale = cosine_similarity(v, vp, dim=-1, keepdim=True)  # (B, h, T, 1)
        if mask is None:
            mask = torch.ones_like(scale)
        scale = scale + (1. - mask) * (-1e9)
        weight = nn.Softmax(dim=-2)(scale)
        weighted_value = weight * v

        # weighted_value = merge(weighted_value)
        context_layer = weighted_value.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hhs,)
        attn_out = context_layer.view(new_context_layer_shape)

        # layer norm, feed forword, skip connection, W_out
        h = self.ffn1(attn_out)
        h = self.layernorm(h + x)
        h = self.ffn2(attn_out)
        h = nn.GELU()(h) + x
        return self.w_out(h)