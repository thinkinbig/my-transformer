import torch
import torch.nn as nn
import math


class PositionEncoder(nn.Module):
    """ Position Encoder
    Position Encoder adds positional information to the input sequence.
    ==================
    Parameters:
    ==================
    d_model: the dimension of input/output
    max_seq_length: the maximum length of the sequence
    """

    def __init__(self, d_model: int, max_seq_length: int):
        super(PositionEncoder, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe: [batch_size, max_seq_length, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1), :]
