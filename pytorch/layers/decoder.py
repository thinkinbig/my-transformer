import torch
import torch.nn as nn

from pytorch.modules.multi_head_attn import MultiHeadAttention
from pytorch.modules.position_feed import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    """Decoder Layer
    Decoder Layer is composed of three sublayers: a masked multi-head self-attention mechanism,
    a multi-head attention mechanism and a position-wise fully connected feed-forward network.
    ==================
    Parameters:
    ==================
    d_model: the dimension of input/output
    num_heads: the number of heads in the multi-head attention
    d_ff: the dimension of the intermediate layer
    dropout: the dropout rate
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
