import torch
import torch.nn as nn
import math


"""Closest Points Algorithm
Split the weight matrix of a linear layer into multiple small linear layers.
==================
Parameters:
==================
linear_layer: the linear layer to be splitted
n_chunks: the number of small linear layers
==================
Return:
==================
a list of small linear layers
"""


def closest_points(linear_layer: torch.nn.Linear, n_chunks: int) -> list:
    in_features, out_features = linear_layer.weight.shape
    chunk_size = out_features // n_chunks
    chunks = torch.chunk(linear_layer.weight, n_chunks, dim=1)
    small_layers = []
    for i in range(n_chunks):
        small_layer = torch.nn.Linear(
            in_features, chunk_size, bias=linear_layer.bias is not None)
        small_layer.weight = torch.nn.Parameter(chunks[i])
        if linear_layer.bias is not None:
            small_layer.bias = torch.nn.Parameter(
                linear_layer.bias[i * chunk_size:(i + 1) * chunk_size])
        small_layers.append(small_layer)
    return small_layers


class MultiHeadAttention(nn.Module):

    """Multi-Head Attention module
    Multi-Head Attention consists of four parts:
    - linear projections
    - scaled dot-product attention
    - concatenation of heads
    - a final linear projection.
    it focuses on the relationship between different words in a sentence.
    ==================
    Parameters:
    ==================
    d_model: the dimension of keys/values/queries
    h: the number of heads
    dropout: the dropout rate
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # linear projections of keys, values, queries and output of attention respectively
        # the output of attention is the concatenation of all heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    """Scaled Dot-Product Attention
    Scale Dot-Product Attention is the core of Multi-Head Attention.
    It determines how much focus should be put on each word in a sentence.
    ==================
    Parameters:
    ==================
    query: the query matrix shape: batch_size * seq_len * d_model
    key: the key matrix shape: batch_size * seq_len * d_model
    value: the value matrix shape: batch_size * seq_len * d_model
    mask: the mask matrix to prevent attention from looking at padding tokens (see the paper for more details) shape: batch_size * seq_len * seq_len
    ==================
    Return:
    ==================
    the output of attention shape: batch_size * seq_len * d_model
    """

    def attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attn_scores = torch.matmul(
            query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask the padding tokens
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        return torch.matmul(attn_probs, value)

    """Split Heads
    Split the keys, values and queries into multiple heads.
    ==================
    Parameters:
    ==================
    x : the input matrix shape: batch_size * seq_len * d_model
    ==================
    Return:
    ==================
    splited heads shape: batch_size * h * seq_len * d_k
    """

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

    """Concat Heads
    Concatenate the heads into a single matrix.
    ==================
    Parameters:
    ==================
    x: the input matrix shape: batch_size * h * seq_len * d_k
    ==================
    Returns:
    ==================
    concatenated heads shape: batch_size * seq_len * d_model
    """

    def concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, d_model = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.h * self.d_k)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        Q, K, V = self.split_heads(self.W_q(query)), self.split_heads(self.W_k(key)), self.split_heads(self.W_v(value))
        mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1) if mask is not None else None
        attn_out = self.attn(Q, K, V, mask)
        return self.W_o(self.concat_heads(attn_out))
