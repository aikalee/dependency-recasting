import torch
import torch.nn as nn

import math
from typing import List, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class FeatureEmbedding(nn.Module):
    """
    Combine 3 features into one embedding:
      - token
      - left bracket list
      - right bracket list

    left/right are pooled with mean.
    Final embedding = token_vec + left_vec + right_vec
    """

    def __init__(
        self,
        vocab_size: int,
        num_left_labels: int,
        num_right_labels: int,
        d_model: int,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.left_emb = nn.Embedding(num_left_labels, d_model)
        self.right_emb = nn.Embedding(num_right_labels, d_model)

        self.d_model = d_model
        self.pad_token_id = pad_token_id

    def _pool_mean(
        self,
        emb_table: nn.Embedding,
        ids_nested: List[List[List[int]]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        ids_nested: B x T x variable_length
        return: [B, T, D]
        """
        batch_vecs = []

        for sent_ids in ids_nested:
            token_vecs = []
            for ids in sent_ids:
                if len(ids) == 0:
                    token_vecs.append(torch.zeros(self.d_model, device=device))
                else:
                    ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
                    vecs = emb_table(ids_tensor)      # [k, D]
                    token_vecs.append(vecs.mean(dim=0)) # mean between tokens
            batch_vecs.append(torch.stack(token_vecs, dim=0))  # [T, D]

        return torch.stack(batch_vecs, dim=0)  # [B, T, D]

    def forward(
        self,
        input_ids: torch.Tensor,
        left_ids: List[List[List[int]]],
        right_ids: List[List[List[int]]],
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        left_ids: B x T x variable_length
        right_ids: B x T x variable_length
        """
        device = input_ids.device

        token_vec = self.token_emb(input_ids)  # [B, T, D]
        left_vec = self._pool_mean(self.left_emb, left_ids, device)    # [B, T, D]
        right_vec = self._pool_mean(self.right_emb, right_ids, device) # [B, T, D]

        return token_vec + left_vec + right_vec # sum between features


class StructuredTokenTransformer(nn.Module):
    """
    Dual-head Transformer for per-token multi-label prediction:
      - left_head: predict left bracket set
      - right_head: predict right bracket set
    """

    def __init__(
        self,
        vocab_size: int,
        num_left_labels: int,
        num_right_labels: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        max_len: int = 2048,
        use_two_layer_head: bool = True,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.num_left_labels = num_left_labels
        self.num_right_labels = num_right_labels

        self.feature_emb = FeatureEmbedding(
            vocab_size=vocab_size,
            num_left_labels=num_left_labels,
            num_right_labels=num_right_labels,
            d_model=d_model,
            pad_token_id=pad_token_id,
        )

        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        if use_two_layer_head:
            self.left_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_left_labels),
            )
            self.right_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_right_labels),
            )
        else:
            self.left_head = nn.Linear(d_model, num_left_labels)
            self.right_head = nn.Linear(d_model, num_right_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        left_ids: List[List[List[int]]],
        right_ids: List[List[List[int]]],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        input_ids:      [B, T]
        left_ids:       B x T x variable_length
        right_ids:      B x T x variable_length
        attention_mask: [B, T], 1 for real token, 0 for padding
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        x = self.feature_emb(input_ids, left_ids, right_ids)  # [B, T, D]
        x = x * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.emb_dropout(x)

        hidden_states = self.encoder(
            x,
            src_key_padding_mask=(attention_mask == 0),
        )

        left_logits = self.left_head(hidden_states)    # [B, T, L]
        right_logits = self.right_head(hidden_states)  # [B, T, R]

        return {
            "left_logits": left_logits,
            "right_logits": right_logits,
        }