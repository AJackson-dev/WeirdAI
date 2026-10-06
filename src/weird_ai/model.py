import torch
import torch.nn as nn

from weird_ai.layer_norm import LayerNorm
from weird_ai.transformer import TransformerBlock


class WeirdAIModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length=128,
        emb_dim=256,
        num_layers=4,
        num_heads=1,
        dropout=0.1,
        qkv_bias=False,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                emb_dim=emb_dim,
                context_length=context_length,
                num_heads=num_heads,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        _, num_tokens = x.shape

        token_embeddings = self.token_embedding(x)
        positions = torch.arange(num_tokens, device=x.device)
        position_embeddings = self.position_embedding(positions)

        x = self.dropout(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
