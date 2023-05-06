import torch
import torch.nn as nn
from pytorch.layers.decoder import DecoderLayer
from pytorch.layers.encoder import EncoderLayer
from pytorch.modules.position_encoder import PositionEncoder


def generate_mask(seq: torch.Tensor):
    mask = (seq != 0).unsqueeze(-2)
    return mask


class Transformer(nn.Module):
    def __init__(self, d_src_vocab: int, d_tgt_vocab: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(d_src_vocab, d_model)
        self.tgt_embed = nn.Embedding(d_tgt_vocab, d_model)
        self.position_enc = PositionEncoder(d_model, max_seq_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, d_tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None):
        if tgt_mask is None:
            tgt_mask = generate_mask(tgt_seq)
        if src_mask is None:
            src_mask = generate_mask(src_seq)
        src_seq = self.dropout(self.position_enc(self.src_embed(src_seq)))
        tgt_seq = self.dropout(self.position_enc(self.tgt_embed(tgt_seq)))

        for enc_layer in self.enc_layers:
            src_seq = enc_layer(src_seq, src_mask)

        for dec_layer in self.dec_layers:
            tgt_seq = dec_layer(tgt_seq, src_seq, tgt_mask, src_mask)
        return self.fc(tgt_seq)
