from pylab import *
import math
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=21, patch_size=7, num_hiddens=512, use_cnn=False):
        super().__init__()

        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x

        self.img_size, self.patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (
            self.img_size[1] // self.patch_size[1]
        )
        self.use_cnn = use_cnn
        self.num_hiddens = num_hiddens

        if self.use_cnn:
            self.conv = nn.LazyConv2d(
                num_hiddens, kernel_size=self.patch_size, stride=self.patch_size
            )
        else:
            self.proj = nn.Linear(self.patch_size[0] * self.patch_size[1] , num_hiddens)

    def forward(self, X):
        if self.use_cnn:
            # Output shape: (batch size, num_patches, num_hiddens)
            return self.conv(X).flatten(2).transpose(1, 2)
        else:
            batch_size, channels, height, width = X.shape
            # Extract patches using unfold
            patches = X.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(3, self.patch_size[1], self.patch_size[1])
            patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size[0] * self.patch_size[1])
            patches = patches.permute(0, 2, 1, 3).reshape(batch_size, 3*49, self.patch_size[0] * self.patch_size[1])
            return self.proj(patches)



def masked_softmax(X, valid_lens):  # @save
    """Perform softmax operation by masking elements on the last axis."""

    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = (
            torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
            < valid_len[:, None]
        )
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):  # @save
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):  # @save
    """Multi-head attention."""

    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads.

        Defined in :numref:`sec_multihead-attention`"""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv.

        Defined in :numref:`sec_multihead-attention`"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)


class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))


class ViTBlock(nn.Module):
    def __init__(
        self,
        num_hiddens,
        norm_shape,
        mlp_num_hiddens,
        num_heads,
        dropout,
        use_bias=False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))



class Encoder(torch.nn.Module):
    """Vision Transformer."""

    def __init__(
        self,
        img_size,
        patch_size,
        num_hiddens,
        mlp_num_hiddens,
        num_heads,
        num_blks,
        emb_dropout,
        blk_dropout,
        lr=0.1,
        use_bias=False,
        num_classes=64,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        
        # Correct number of steps for positional embedding
        num_steps = 3*49 + 1  # 27 patches + 1 class token = 28 positions
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(
                f"{i}",
                ViTBlock(
                    num_hiddens,
                    num_hiddens,
                    mlp_num_hiddens,
                    num_heads,
                    blk_dropout,
                    use_bias,
                ),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, num_classes)
        )

    def forward(self, X):
        X = self.patch_embedding(X)  # Output shape: [batch_size, 27, num_hiddens]
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)  # Adding the class token, now X.shape[1] = 28
        
        # Ensure positional embedding size matches
        X = self.dropout(X + self.pos_embedding[:, :X.shape[1], :])  
        
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
