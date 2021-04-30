import math
import torch
from torch import nn
from torch.nn import functional as F

from fast_transformers.attention import CausalLinearAttention
from fast_transformers.attention.causal_linear_attention import causal_linear
from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.feature_maps import elu_feature_map


def get_positional_encoding(position, channels, min_timescale=1.0, max_timescale=1.0e4):
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=position.dtype, device=position.device) * -log_timescale_increment)
    scaled_time = position * inv_timescales.view(1, 1, -1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], -1)
    signal = F.pad(signal, [0, channels % 2])
    return signal


class ImageEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb = nn.Linear(in_channels, out_channels)

    def forward(self, x, x_mask, pos_r, pos_c):
        half_channels = self.out_channels // 2
        x = self.emb(x)
        x_pos_r = get_positional_encoding(pos_r, half_channels)
        x_pos_c = get_positional_encoding(pos_c, self.out_channels - half_channels)
        x_pos = torch.cat([x_pos_r, x_pos_c], -1)
        x_emb = (x + x_pos) * x_mask
        return x_emb


class TextEmbedding(nn.Module):
    def __init__(self, n_vocab, out_channels):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.emb = nn.Embedding(n_vocab, out_channels)

    def forward(self, x, x_mask, pos_t):
        x = self.emb(x)
        x_pos_t = get_positional_encoding(pos_t, self.out_channels)
        x_emb = (x + x_pos_t) * x_mask
        return x_emb


class CausalLinearAttentionAMP(CausalLinearAttention):

    def forward(self, queries, keys, values, query_mask=None,
                key_mask=None, cache=None):
        self.feature_map.new_feature_map()
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        if cache is not None:
            with torch.cuda.amp.autocast(enabled=False):
                Q = Q.float() # [b, t, nh ,d]
                K = K.float() # [b, t, nh, d]
                if key_mask is not None:
                    K = K * key_mask[:, :, :, None]

                values = values.float() # [b, t, nh, d]
                Q_p = Q.unsqueeze(-2) # [b, t, nh, 1, d]
                K_p = K.unsqueeze(-1) # [b, t, nh, d, 1]
                values_p = values.unsqueeze(-2) # [b, t, nh, 1, d]
                kv_cum = cache['kv'] + (K_p * values_p).cumsum(1) # [b, t, nh, d, d]
                K_cum = cache['k_cum'] + K.cumsum(1) # [b, t, nh, d]
                cache['kv'] = kv_cum[:,-1:] # [b, 1, nh, d, d]
                cache['k_cum'] = K_cum[:,-1:] # [b, 1, nh, d]

                V = (Q_p @ kv_cum).squeeze(-2) # [b, t, nh, d]
                Z = 1/(torch.sum(Q * K_cum, -1) + self.eps) # [b, t, nh, d], [b, t, nh, d]
                out = V * Z[:, :, :, None] # [b, t, nh, d], [b, t, nh, 1]
                out = out.to(dtype=queries.dtype)
                return out
        else:
            K = K * key_mask[:, :, :, None]
            Q, K = self._make_sizes_compatible(Q, K)

            with torch.cuda.amp.autocast(enabled=False):
                Q = Q.float()
                K = K.float()
                values = values.float()

                # Compute the normalizers
                Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
                if getattr(self, "save_attn", False):
                    self.attn_map = (Q.permute(0,2,1,3).contiguous() @ K.permute(0,2,3,1).contiguous()).tril()*Z.permute(0,2,1).unsqueeze(-1)

                # Compute the unnormalized result
                V = causal_linear(
                    Q,
                    K,
                    values
                )
                out = V * Z[:, :, :, None]
            out = out.to(dtype=queries.dtype)
        return out

    def set_save_attn(self, v):
        self.save_attn = v
    
    
class AttentionLayer(nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super().__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, query_mask=None,
                key_mask=None, cache=None):
        """Apply attention to the passed in queries/keys/values after
        projecting them to multiple heads.

        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            query_mask,
            key_mask,
            cache=cache
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)

    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu"):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_mask=None, cache=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
        """
        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            query_mask=x_mask,
            key_mask=x_mask,
            cache=cache
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)


class CausalLinearTransformerEncoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, 
        p_dropout=0.1, activation="gelu", feature_map=elu_feature_map):

        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                AttentionLayer(
                    CausalLinearAttentionAMP(hidden_channels, feature_map),
                    hidden_channels,
                    n_heads),
                hidden_channels,
                filter_channels,
                p_dropout,
                activation
            )
            for i in range(n_layers)
        ])

    def forward(self, x, x_mask=None, cache=None):
        cache_l = None
        
        # Apply all the transformers
        for i, layer in enumerate(self.layers):
            if cache is not None:
                cache_l = cache[i]
            x = layer(x, x_mask=x_mask, cache=cache_l)

        return x


class TableRecognizer(nn.Module):
    def __init__(self, n_vocab, img_channels, hidden_channels, filter_channels, n_heads, n_layers, p_dropout=.1):
        super().__init__()
        self.n_vocab = n_vocab
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.emb_img = ImageEmbedding(img_channels, hidden_channels)
        self.emb_txt = TextEmbedding(n_vocab, hidden_channels)
        self.enc = CausalLinearTransformerEncoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            p_dropout)

        self.proj_img = nn.Linear(hidden_channels, img_channels)
        self.proj_txt = nn.Linear(hidden_channels, n_vocab)

    def forward(self, x_img, x_txt, mask_img, mask_txt, pos_r, pos_c, pos_t):
        x_mask = mask_img + mask_txt

        x_emb_img = self.emb_img(x_img, mask_img, pos_r, pos_c)
        x_emb_txt = self.emb_txt(x_txt, mask_txt, pos_t)
        x_emb = x_emb_img + x_emb_txt

        x = self.enc(x_emb, x_mask)
        logit_img = self.proj_img(x)
        logit_txt = self.proj_txt(x)
        return logit_img, logit_txt

    def inference(self, x_img, mask_img, pos_r, pos_c, idx_start=1, idx_end=2, max_decode_len=10000):
        from tqdm import tqdm
        with torch.no_grad():
            b = x_img.size(0)
            nh = self.n_heads
            d = self.hidden_channels // self.n_heads
            dtype = x_img.dtype
            device = x_img.device

            cache = [{
                "kv": torch.zeros(b, 1, nh, d, d).to(dtype=torch.float, device=device),
                "k_cum": torch.zeros(b, 1, nh, d).to(dtype=torch.float, device=device)
                } for _ in range(self.n_layers)
            ]
            x_emb_img = self.emb_img(x_img, mask_img, pos_r, pos_c)
            _ = self.enc(x_emb_img, mask_img, cache)

            pos_enc = get_positional_encoding(
                torch.arange(max_decode_len).view(1,-1,1).to(device=device), 
                self.hidden_channels
            )
            finished = torch.BoolTensor(b,1).to(device=device).fill_(False)
            idx = torch.zeros(b,1).long().to(device=device) + idx_start
            ids = []
            for i in tqdm(range(max_decode_len)):
                x_emb_txt = self.emb_txt.emb(idx) + pos_enc[:,i:i+1]
                x = self.enc(x_emb_txt, None, cache)
                logit_txt = self.proj_txt(x)
                idx = torch.argmax(logit_txt, -1)
                ids.append(idx)
                finished |= torch.eq(idx, idx_end)
                if torch.all(finished):
                    break
            return ids
