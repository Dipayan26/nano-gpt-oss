import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import torch
import torch.nn as nn

print(torch.__version__)  # e.g., '2.0.1+cu118'

vocab_size = 50000
embed_dim = 10

embedding_layer = nn.Embedding(vocab_size, embed_dim)
print(embedding_layer.weight.shape)  # torch.Size([50000, 768])


embedding_layer.weight[0]  # shape (768,)
embedding_layer.weight[1]  # shape (768,)
embedding_layer.weight[2]  # shape (768,)

embedding_layer.weight[49000]  # shape (768,)
embedding_layer.weight[50000]  # shape (768,)



import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from dataclasses import dataclass

help(dataclass)

#_______________________________________________________________
#_______________________________________________________________
@dataclass
class ModelConfig:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


config = ModelConfig(hidden_size=64)

config.rope_scaling_factor = 16.0

class TransformerModel(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size
            )
            for _ in range(config.num_hidden_layers)
        ])
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return x


config = ModelConfig(hidden_size=1024, num_hidden_layers=12)
model = TransformerModel(config)

#_______________________________________________________________
#_______________________________________________________________


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None):
        
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


norm = RMSNorm(num_features=2)


a= torch.randn( 5, 2)  # (batch_size, seq_length , hidden_size)
x = a.float()
a.dtype
x.dtype
a**2


scale = torch.ones(2, dtype=torch.float32)

## dim -1 means last dimension row wise mean will be calculated  
eps = torch.mean(a**2, dim=-1, keepdim=True)

t = torch.tensor([2,3]).float()

eps= 1e-05
t = (torch.mean(t**2, dim=-1, keepdim=True) )
t = torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) )
t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) )

t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
(t * scale)


t, dtype = a.float(), a.dtype
a.shape[-1]
output = norm(a)

###_______________________________________________________________
###_______________________________________________________________
@dataclass
class ModelConfig:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)



class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


## sliding window attention with sinks
def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1) # ##torch.Size([5, 8, 2, 64]) same as query by expanding means repeating the tensor along that dimension
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1) ##torch.Size([5, 8, 2, 64]) 
    
    #sinks
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1) 
    
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )

    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)## batch matrix multiplication query and key transposed
    
    QK *= sm_scale # scaling
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1) #S is added to the last dimension as sink tokens
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1] # removing the sink token weights for final attention calculation
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)## final attention calculation
    return attn.reshape(n_tokens, -1)



sm_scale = 1 / math.sqrt(ModelConfig.head_dim)# hed dim is 64

layer_idx = 2
sliding_window = ModelConfig.sliding_window if layer_idx % 2 == 0 else 0

#head is 8
q = torch.randn( 5, 8, 2, 64)  # (n_tokens, n_heads, q_mult, d_head)
## q_mult means -- per head there is 2 query vectors
k = torch.randn( 5, 8, 64)  # (n_tokens, n_heads, d_head)
v = torch.randn( 5, 8, 64)  # (n_tokens, n_heads, d_head)
sinks = torch.randn(8, 2, 1, 1)  #(n_heads, q_mult, 1, 1)  


#_______________________________________________________________
q_mult = 2
k11 = torch.randn( 5, 8, 64)  # (n_tokens, n_heads, d_head)

K1 = k11[:, :, None, :].expand(-1, -1, q_mult, -1)
K1.shape ##torch.Size([5, 8, 2, 64]) 
## expand means -- it will repeat the tensor along that dimension
#_______________________________________________________________
sinks = torch.randn(8, 2, 1, 1)  #(n_heads, q_mult, 1, 1)  
n_heads = 8
n_tokens = 5
S = sinks.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
S.shape  ##torch.Size([8, 2, 5, 1])
#_______________________________________________________________
'''the variable q is not used for its values.
Its used only as a template — meaning it provides metadata (like dtype, device, and sometimes layout).'''
q1 = q.new_full((n_tokens, n_tokens),1)
q1.shape
q1 = q.new_full((n_tokens, n_tokens),-float("inf"))
torch.triu(q1, diagonal=1)
mask = torch.triu(q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
#_______________________________________________________________
tensor = torch.ones((2,), dtype=torch.float64)
tensor.new_full((3, 4), 3.141592)
sliding_window = 2 # sliding window in real code is 128 it stays same

mask.new_full((n_tokens, n_tokens), -float("inf"))
torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-1)

if sliding_window > 0:
    mask += torch.tril(
        mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
    )
#_______________________________________________________________
q = torch.randn( 5, 8, 2, 4)  # (n_tokens, n_heads, q_mult, d_head)
k11 = torch.randn( 5, 8, 4)  # (n_tokens, n_heads, d_head)

K1 = k11[:, :, None, :].expand(-1, -1, q_mult, -1)
K1.shape ##torch.Size([5, 8, 2, 64]) 

# Q*Ktranspose = (5,8,2,64) * (5,8,2,64) --> (8,2,5,5)
# QK = torch.einsum("qhmd,khmd->hmqk", q, K1)# making directly as head first
#Any letter that appears in both inputs but not in the output is summed over (contracted).
QK = torch.einsum("qhmd,khmd->hmqk", q, K1)
# QK = torch.einsum("qhmd,khmd->qhmd", q, K1)
# QK = torch.einsum("qhmd,khmd->qhmk", q, K1)
QK.shape  ##torch.Size([8, 2, 5, 5])

#_______________________________________________________________
QK *= sm_scale # scaling
QK += mask[None, None, :, :]
QK = torch.cat([QK, S], dim=-1) #S is added to the last dimension as sink tokens
W = torch.softmax(QK, dim=-1)
W = W[..., :-1] # removing the sink token weights for final attention calculation
W.shape ##torch.Size([8, 2, 5, 5])
v.shape
v1 = v[:, :, None, :].expand(-1, -1, q_mult, -1)
v1.shape ##torch.Size([5, 8, 2, 64])
attn1 = torch.einsum("hmqk,khmd->qhmd", W, v1)## final attention calculation
attn1.shape ##torch.Size([5, 8, 2, 64])
attn = attn1.reshape(n_tokens, -1)
attn.shape ##torch.Size([5, 8, 2, 64])

#_______________________________________________________________


t = sdpa(q, k, v, sinks, sm_scale, sliding_window)

t.shape


#_______________________________________________________________
#_______________________________________________________________
#_______________________________________________________________
#_______________________________________________________________
@dataclass
class ModelConfig:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8 # no of key value groups
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        #########################################
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16))
        ####################################
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        ## head dim will be = 64+2*8=80
        ## qkv_dim = 80*64=5120
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        #hidden_size: int = 2880, 
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x) ## RMSNorm
        
        qkv = self.qkv(t) ## Linear layer making the query key value together 

        ## spliting the qkv into q , k , v ##############        
        #splitting the Query 
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        #splitting the Key
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        #splitting the Value
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        
        #reshaping the q , k , v
        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        
        # applying rotary embedding to q and k
        q, k = self.rope(q, k)
        # applying sliding window attention with sinks
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        # final linear layer
        t = self.out(t)
        # residual connection
        t = x + t
        return t






a = torch.randn( 5, 2560)  # (batch_size, seq_length , hidden_size)






































































































































