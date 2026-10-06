from torch import nn as nn
from weird_ai.attention import CausalAttention
from weird_ai.feed_forward import FeedForward
from weird_ai.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    # TODO 
    # Create a TransformerBlock class, inheriting from nn.Module
    # using 
    #  - LayerNorm
    #  - SelfAttention from previous assignment
    #  - FeedForward
    #  - Residual connections

    def __init__(self, emb_dim, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.norm1 = LayerNorm(emb_dim)
        self.attention = CausalAttention(
            embedding_dim=emb_dim,
            output_dim=emb_dim,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )

        self.norm2 = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
    pass
