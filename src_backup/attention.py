import math

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    score_i = f(Q, K_i), i = 1, 2, ..., t
        dot:        f(Q, K_i) = Q.transpose · K_i
        scaled_dot: f(Q, K_i) = Q.transpose · K_i / √(key_dim)
        general:    f(Q, K_i) = Q.transpose · W · K_i
        concat:     f(Q, K_i) = V.transpose · tanh(W · [Q; K_i])
        perceptron: f(Q, K_i) = V.transpose · tanh(W · Q + U · K_i)

    alpha_i = softmax(score_i)

    context = Σ(alpha_i · V_i)

    Args:
        query_dim: Dimension of query vector (Q).
        key_dim: Dimension of key vectors (K_i, i = 1, 2, ..., t).
        method: dot/scaled_dot/general/concat/perceptron
    """

    def __init__(self, query_dim, key_dim, value_dim=0, method='general', dropout_rate=0.):
        super(Attention, self).__init__()
        self.method = method
        self.dropout = nn.Dropout(dropout_rate)

        if self.method == 'dot' or self.method == 'scaled_dot':
            assert query_dim == key_dim, "The query_dim must equals key_dim."
            if value_dim == 0:
                value_dim = key_dim

            self.linear_q = nn.Linear(query_dim, query_dim)
            self.linear_k = nn.Linear(key_dim, key_dim)
            self.linear_v = nn.Linear(value_dim, value_dim)

        elif self.method == 'general':
            self.W = nn.Linear(query_dim, key_dim, bias=False)

        elif self.method == 'concat':
            self.W = nn.Linear(query_dim + key_dim, query_dim + key_dim, bias=False)
            self.V = nn.Linear(query_dim + key_dim, 1, bias=False)

        elif self.method == 'perceptron':
            self.W = nn.Linear(query_dim, query_dim, bias=False)
            self.U = nn.Linear(key_dim, query_dim, bias=False)
            self.V = nn.Linear(query_dim, 1, bias=False)

        else:
            raise ValueError('The method must be one of the following: dot, scaled_dot, general, concat or perceptron.')

    def forward(self, queries, keys, values=None, mask=None, top_k=None):
        """
        Args:
            queries: Batch of query vectors (Q). Tensor[batch_size, query_len, query_dim]
            keys: Batch of key vectors (K_i, i = 1, 2, ..., t). Tensor[batch_size, key_len, key_dim]
            values: Batch of value vectors (V_i, i = 1, 2, ..., t). Tensor[batch_size, value_len, value_dim]
            mask: Use none zero value as valid flag and 0 as pad flag. Tensor[batch_size, query_len, key_len]
            top_k: Select top K relative values. int(0, ∞)

        Return:
            Batch of context vector (C). Tensor[batch_size, query_len, value_dim]
        """

        if values is None:
            values = keys
        else:
            assert values.shape[-2] == keys.shape[-2], "value_len Must equals key_len."

        if self.method == 'dot' or self.method == 'scaled_dot':
            queries = self.linear_q(queries)
            keys = self.linear_k(keys)
            values = self.linear_v(values)

        scores = self.score(queries, keys)  # [batch_size, query_len, key_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))  # [batch_size, query_len, key_len]

        alphas = F.softmax(scores, dim=-1)  # [batch_size, query_len, key_len]
        alphas = alphas.masked_fill(alphas != alphas, 0)  # set all 'nan' to 0.

        if top_k is not None:
            _, indices = torch.topk(alphas, k=top_k, dim=-1, largest=True)  # [batch_size, query_len, top_k]
            alphas_device = self.get_device(alphas)
            # [batch_size, query_len, key_len]
            topk_mask = torch.zeros(alphas.shape).to(alphas_device).scatter_(
                dim=-1,
                index=indices,
                src=torch.ones(indices.shape).to(alphas_device)
            )
            alphas = alphas.masked_fill(topk_mask == 0, 0)
            alphas = F.softmax(alphas, dim=-1)  # [batch_size, query_len, key_len]
            alphas = alphas.masked_fill(alphas != alphas, 0)  # set all 'nan' to 0.

        alphas = self.dropout(alphas)

        return torch.bmm(alphas, values)  # [batch_size, query_len, value_dim]

    def score(self, queries, keys):
        """
        Args:
            queries: Tensor[batch_size, query_len, query_dim]
            keys: Tensor[batch_size, key_len, key_dim]

        Return:
            Batch of attention scores. Tensor[batch_size, query_len, key_len]
        """

        if self.method == 'dot' or self.method == 'scaled_dot':
            # f(Q, K_i) = Q.transpose · K_i
            # f(Q, K_i) = Q.transpose · K_i / √(key_dim)
            # queries: [batch_size, query_len, input_dim]
            # keys: [batch_size, key_len, input_dim]
            scores = torch.bmm(queries, keys.transpose(-1, -2))  # [batch_size, query_len, key_len]
            if self.method == 'scaled_dot':
                scores /= math.sqrt(keys.shape[-2])

            return scores  # [batch_size, query_len, key_len]

        elif self.method == 'general':
            # f(Q, K_i) = Q.transpose · W · K_i
            return torch.bmm(self.W(queries), keys.transpose(-1, -2))  # [batch_size, query_len, key_len]

        elif self.method == 'concat':
            # f(Q, K_i) = V.transpose · tanh(W · [Q; K_i])

            # [batch_size, query_len, key_len, query_dim]
            queries = queries.unsqueeze(2).expand(-1, -1, keys.shape[1], -1)
            # [batch_size, query_len, key_len, key_dim]
            keys = keys.unsqueeze(1).expand(-1, queries.shape[1], -1, -1)

            scores = torch.cat([queries, keys], dim=-1)  # [batch_size, query_len, key_len, query_dim + key_dim]
            scores = self.W(scores)  # [batch_size, query_len, key_len, query_dim + key_dim]
            scores = torch.tanh(scores)  # [batch_size, query_len, key_len, query_dim + key_dim]
            return self.V(scores).squeeze(3)  # [batch_size, query_len, key_len]

        elif self.method == 'perceptron':
            # f(Q, K_i) = V.transpose · tanh(W · Q + U · K_i)

            # [batch_size, query_len, key_len, query_dim]
            queries = queries.unsqueeze(2).expand(-1, -1, keys.shape[1], -1)
            # [batch_size, query_len, key_len, key_dim]
            keys = keys.unsqueeze(1).expand(-1, queries.shape[1], -1, -1)

            scores = self.W(queries) + self.U(keys)  # [batch_size, query_len, key_len, query_dim]
            scores = torch.tanh(scores)  # [batch_size, query_len, key_len, query_dim]
            return self.V(scores).squeeze(3)  # [batch_size, query_len, key_len]

    @staticmethod
    def get_device(t):
        try:
            device_id = t.get_device()
        except:
            return 'cpu'
        else:
            return 'cpu' if device_id < 0 else 'cuda'


class MultiHeadAttention(nn.Module):
    """
    Args:
        input_dim: Last dimension of queries, keys, values tensor.
        num_heads: Heads number.
        dropout_rate: Dropout.
    """

    def __init__(self, input_dim, num_heads=8, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.linear_q = nn.Linear(input_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(input_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(input_dim, self.dim_per_head * num_heads)

        self.attn = Attention(query_dim=input_dim,
                              key_dim=input_dim,
                              method='scaled_dot',
                              dropout_rate=dropout_rate)
        self.linear_final = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, queries, keys, values, mask=None):
        """
        Args:
            queries: Tensor[batch_size, query_len, input_dim]
            keys: Tensor[batch_size, key_len, input_dim]
            values: Tensor[batch_size, value_len, input_dim]
            mask: Use none zero value as valid flag and 0 as pad flag. Tensor[batch_size, query_len, key_len]

        Return:
            Batch of context vector (C). Tensor[batch_size, query_len, input_dim]
        """

        residual = queries  # [batch_size, query_len, input_dim]
        batch_size = residual.shape[0]

        # Linear projection.
        queries = self.linear_q(queries)  # [batch_size, query_len, input_dim]
        keys = self.linear_k(keys)  # [batch_size, key_len, input_dim]
        values = self.linear_v(values)  # [batch_size, value_len, input_dim]

        # Split by heads.
        # [batch_size * num_heads, seq_len, dim_per_head]
        queries = queries.view(batch_size * self.num_heads, -1, self.dim_per_head)
        keys = keys.view(batch_size * self.num_heads, -1, self.dim_per_head)
        values = values.view(batch_size * self.num_heads, -1, self.dim_per_head)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)  # [batch_size * num_heads, query_len, key_len]

        # Scaled-dot product attention.
        # [batch_size * num_heads, query_len, dim_per_head]
        context = self.attn(queries=queries, keys=keys, values=values, mask=mask)
        # [batch_size, query_len, input_dim]
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)

        # Final linear projection.
        context = self.linear_final(context)  # [batch_size, query_len, input_dim]

        # Dropout.
        context = self.dropout(context)  # [batch_size, query_len, input_dim]

        # Residual and batch norm.
        return self.layer_norm(residual + context)  # [batch_size, query_len, input_dim]
