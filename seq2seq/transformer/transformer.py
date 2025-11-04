import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        pad_idx: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        max_length: int,
        value_length: int,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length
        self.device = device

        self.encoder = Encoder(
            vocab_size,
            num_layers,
            num_heads,
            embedding_dim,
            ffn_hidden_dim,
            qk_length,
            value_length,
            max_length,
            dropout,
        )

        self.decoder = Decoder(
            vocab_size,
            num_layers,
            num_heads,
            embedding_dim,
            ffn_hidden_dim,
            qk_length,
            value_length,
            max_length,
            dropout,
        )

    def make_pad_mask(self, q: torch.Tensor, k: torch.Tensor):
        pad_mask = k.eq(self.pad_idx).unsqueeze(1).unsqueeze(1)
        return pad_mask.to(self.device)

    def make_causal_mask(self, q: torch.Tensor, k: torch.Tensor):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.triu(
            torch.ones(len_q, len_k, device=self.device, dtype=torch.bool), diagonal=1
        )
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # print("STARTING")

        # print(src.size())

        src_mask = self.make_pad_mask(src, src)
        src_tgt_mask = self.make_pad_mask(tgt, src)

        tgt_pad_mask = self.make_pad_mask(tgt, tgt)
        tgt_causal_mask = self.make_causal_mask(tgt, tgt)

        tgt_mask = tgt_pad_mask | tgt_causal_mask


        # print("ENCODER")
        # print(src.size())
        enc_src = self.encoder(src, src_mask)
        # print(src.size())
        # print(enc_src.size())
        # print("DECODER")
        output = self.decoder(tgt, enc_src, tgt_mask, src_tgt_mask)
        # print(output.size(), src.size())
        # print("DONE")

        return output
