import torch
from torch import nn

from src.models.activations import GeLU
from src.models.attention import MultiHeadAttention
from src.models.layer_normalization import LayerNorm


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], 4 * config['emb_dim']),
            GeLU(),
            nn.Linear(config['emb_dim'] * 4, config['emb_dim'])
        )

    def forward(self, x):
        return self.layers(x)



class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        # layer normalization
        self.layer_norm1 = LayerNorm(config['emb_dim'])

        # attention
        self.multi_head_attention = MultiHeadAttention(
            d_in=config['emb_dim'],
            d_out=config['emb_dim'],
            context_length=config['context_length'],
            num_heads=config['n_heads'],
            dropout=config['drop_rate'],
            bias=config['qkv_bias']
        )

        # dropout
        self.dropout1 = nn.Dropout(p=config['drop_rate'])

        # layer normalization 2
        self.layer_norm2 = LayerNorm(config['emb_dim'])

        # feed forward
        self.feed_forward = FeedForward(config)

        # dropout
        self.dropout2 = nn.Dropout(p=config['drop_rate'])

    def forward(self, x):
        output = self.layer_norm1(x)
        output = self.multi_head_attention(output)
        output = self.dropout1(output)
        output = output + x

        x2 = output

        output = self.layer_norm2(output)
        output = self.feed_forward(output)
        output = self.dropout2(output)
        output = output + x2

        return output
