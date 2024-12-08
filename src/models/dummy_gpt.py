import torch
from torch import nn
from src.models.attention import MultiHeadAttention

class DummyTransformersBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        # embedding layer
        self.tok_emb = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.pos_emb = nn.Embedding(config['context_length'], config['emb_dim'])
        self.drop_emb = nn.Dropout(config['drop_rate'])

        # transformers block
        self.trf_blocks = nn.Sequential(
            * [ DummyTransformersBlock(config) for _ in range(config['n_layers']) ]
        )

        # Normalization layer
        self.final_norm = DummyLayerNorm(config['emb_dim'])

        # projection layer
        self.out_head = nn.Linear(
            config['emb_dim'],
            config['vocab_size'],
            bias=False
        )


    def forward(self, in_idx):

        # getting the batch size and sequence length
        batch_size, seq_len = in_idx.shape

        # embedding
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # transformers bloc
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits








