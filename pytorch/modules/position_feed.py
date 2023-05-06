import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Networks
    Position-wise Feed-Forward Networks caculates the output of each position separately
    using a two-layer feed-forward network.
    ==================
    Parameters:
    ==================
    d_model: the dimension of input/output
    d_ff: the dimension of the intermediate layer
    dropout: the dropout rate
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
