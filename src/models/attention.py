import torch
from torch import nn


torch.manual_seed(77)

class SelfAttention_v1(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()

        # initialize weights
        self.W_Q = torch.nn.Parameter(torch.randn(embedding_dim, output_dim), requires_grad=False)
        self.W_K = torch.nn.Parameter(torch.randn(embedding_dim, output_dim), requires_grad=False)
        self.W_V = torch.nn.Parameter(torch.randn(embedding_dim, output_dim), requires_grad=False)


    def forward(self, inputs):
        """
        Forward pass
        :param inputs: input tensor of shape (batch_size, seq_length, embedding_dim)
        :return:
        """

        queries = inputs @ self.W_Q
        keys = inputs @ self.W_K
        values = inputs @ self.W_V

        # calculate the attentions scores
        attention_scores = queries @ keys.T
        value_dim = values.shape[-1]
        attention_weights = torch.softmax(
            attention_scores  /  value_dim ** 0.5, dim=-1
        )
        context_vec = attention_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, embedding_dim: int, output_dim: int, attention_bias: bool = False):
        super().__init__()

        # initialize weights
        self.W_Q = torch.nn.Linear(embedding_dim, output_dim, bias=attention_bias)
        self.W_K = torch.nn.Linear(embedding_dim, output_dim, bias=attention_bias)
        self.W_V = torch.nn.Linear(embedding_dim, output_dim, bias=attention_bias)


    def forward(self, inputs):
        """
        Forward pass
        :param inputs: input tensor of shape (batch_size, seq_length, embedding_dim)
        :return:
        """

        queries = self.W_Q(inputs)
        keys = self.W_K(inputs)
        values = self.W_V(inputs)

        # calculate the attentions scores
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores  /  keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec = attention_weights @ values
        return context_vec


class MultiHeadAttention(nn.Module):
    """
    Multi head attention layer
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int, context_length: int, dropout: float = 0.1, bias: bool = True):
        
        super().__init__()

        # ensure the output dim is divisible by the number of heads
        assert d_out % num_heads == 0, 'Output dimension must be divisible by the number of heads'

        self.head_dim = d_out // num_heads
        self.num_heads = num_heads
        self.d_out = d_out
        self.d_in = d_in
        self.context_length = context_length

        self.W_Q = torch.nn.Linear(d_in, d_out, bias=bias)
        self.W_K = torch.nn.Linear(d_in, d_out, bias=bias)
        self.W_V = torch.nn.Linear(d_in, d_out, bias=bias)

        self.out_proj = torch.nn.Linear(d_out, d_out, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):

        batch_size, seq_length, d_in = x.shape

        keys = self.W_K(x)
        queries = self.W_Q(x)
        values = self.W_V(x)

        # reshape to the right format
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # calculate the attentions scores
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1)
        mask_bool = self.mask[:seq_length, :seq_length].bool()
        attn_scores = attn_scores.masked_fill(mask_bool, float('-inf'))
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, seq_length, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


