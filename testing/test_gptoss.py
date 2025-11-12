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
        
        ## qkv_dim = 64 * (64+(2*8)) = 5120
        #2 means for key and value
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads)
        #hidden_size: int = 2880, 
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        '''
        Essentially, each input vector (of dimension hidden_size, say 2880) is transformed into Q, K, and V vectors with separate weights — but all packed into one linear layer for computational efficiency.
        Input dimension:______________________________
        config.hidden_size → e.g. 2880
        This is the size of the input embedding per token (coming from the previous layer or embedding layer).
        Output dimension:_______________________________
        qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        For example, with head_dim=64, num_attention_heads=64, and num_key_value_heads=8:
        qkv_dim = 64 * (64 + 2*8) = 64 * 80 = 5120
        This means the linear layer outputs a vector of size 5120 for each input token.
        So, it outputs a concatenated tensor containing all Q, K, and V vectors.
        '''
        #________________________________________________________________
        
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
        print(' after norm t shape: ', t.shape)
        qkv = self.qkv(t) ## Linear layer making the query key value together 
        print(' after qkv linear layer shape: ', qkv.shape)
        ###############
        ## spliting the qkv into q , k , v ##############    
        '''
        #|<---- Q (4096) ---->|<-- K (512) -->|<-- V (512) -->| 
        Q → all 64 query heads × 64 dim = 4096
        K → 8 key heads × 64 dim = 512
        V → 8 value heads × 64 dim = 512
        '''   
        
        #splitting the Query 
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        print(' after splitting q shape: ', q.shape)
        '''
        #5120-4096 =1024 
            # 1024/2 = 512 for key and 512 for value each
            q = qkv[:, : ModelConfig.num_attention_heads * ModelConfig.head_dim]
            q = qkv[:, :4096]
            You just pointed to a segment of the full tensor, not copied it.
            So its stride info still refers to the original 5120-wide layout — non-contiguous.

            .contiguous() copies those 4096 entries per row into a new buffer laid out perfectly:
            q = qkv[:, : ModelConfig.num_attention_heads * ModelConfig.head_dim].contiguous()
        '''
        #splitting the Key
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        print(' after splitting k shape: ', k.shape)
        '''
        self.num_attention_heads * self.head_dim --> 64 * 64 = 4096 to 
        (self.num_attention_heads + self.num_key_value_heads) * self.head_dim--> (64 + 8) * 64 = 72 * 64 = 4608

        '''
        
        #splitting the Value
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        print(' after splitting v shape: ', v.shape)
        '''
            (self.num_attention_heads + self.num_key_value_heads)* self.head_dim --> (64 + 8) * 64 = 72 * 64 = 4608 to (this portion is from before )
            and (self.num_attention_heads + 2 * self.num_key_value_heads)* self.head_dim --> (64 + 16) * 64 = 80 * 64 = 5120
            5120 is the total length of qkv   
        '''
        
        #reshaping the q , k , v
        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        print(' after reshaping q shape: ', q.shape)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        print(' after reshaping k shape: ', k.shape)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        print(' after reshaping v shape: ', v.shape)
        
        '''
        >>> a = torch.randn( (5, 2880), dtype=torch.bfloat16) # (batch_size, seq_length , hidden_size)
        >>> a.dtype
        torch.bfloat16
        >>> block = AttentionBlock(ModelConfig(), layer_idx=2)
        >>> out = block(a)
        after norm t shape:  torch.Size([5, 2880])
        after qkv linear layer shape:  torch.Size([5, 5120])
        after splitting q shape:  torch.Size([5, 4096])
        after splitting k shape:  torch.Size([5, 512])
        after splitting v shape:  torch.Size([5, 512])
        
        after reshaping q shape:  torch.Size([5, 8, 8, 64])
        --> so there is 5 tokens , 8 key value groups , each group has 8 heads , each head has 64 dimension
        after reshaping k shape:  torch.Size([5, 8, 64]), we have different key for each group and each head in that group
        --> so there is 5 tokens , 8 key value groups , each group has 64 dimension and we are sharing the key across all heads in that group , means we have same key sharing all the heads in a group. 
        after reshaping v shape:  torch.Size([5, 8, 64])
        --> same as key 
        >>> out.shape
        torch.Size([5, 2880])
        >>>

        '''
        # applying rotary embedding to q and k
        q, k = self.rope(q, k)
        # applying sliding window attention with sinks
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        # final linear layer
        t = self.out(t)
        # residual connection
        t = x + t
        return t



a = torch.randn( (5, 2880), dtype=torch.bfloat16) # (batch_size, seq_length , hidden_size)
a.dtype
block = AttentionBlock(ModelConfig(), layer_idx=2)
out = block(a)
out.shape

qkv=  torch.randn(5, 5120)  # (batch_size, seq_length , qkv_dim)
#ModelConfig.num_attention_heads * ModelConfig.head_dim == 64 * 64 = 4096


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

#____________________________________________________________



#-- MLP with SWIGLU activation and Mixture of Experts --
def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    '''
    2. Why this splitting?
        This pattern is used in Gated Linear Units (GLUs) like SiLU/Swish-Gated Linear Units (SiGLU or SwiGLU) — a modern activation function used in transformer MLPs (e.g., LLaMA, PaLM, Mistral, etc.).
        When the MLP projection produces a tensor x of size (..., 2 * hidden_size),
        the idea is that:
        the first half (even indices) acts as the gating signal (like a sigmoid gate),
        the second half (odd indices) acts as the linear signal.
    '''
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    '''
    clamp() limits (or clips) the values in a tensor so they don’t go below or above certain thresholds.
        min → sets a lower bound (values smaller than this become equal to it)
        max → sets an upper bound (values larger than this become equal to it)
    '''
    x_linear = x_linear.clamp(min=-limit, max=limit)
    
    out_glu = x_glu * torch.sigmoid(alpha * x_glu) #This is performing a gated nonlinearity, similar to Swish or SiLU, used in SwiGLU (Swish-Gated Linear Unit).
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1) #So this line multiplies the gate (out_glu) with the content (x_linear + 1).
'''
    Think of it like this:
        x_glu = how open the faucet is (0 = closed, 1 = fully open)
        x_linear = the water flow intensity
        Adding +1 = ensures there’s always at least a trickle of flow even when the gate is weak
        This keeps information flowing even when the model hasn’t yet learned meaningful activations — which helps early training stability.
'''

x = torch.Tensor([[0.5, 1.0, 2.0, 13.0],
                  [4.0, 10.0, 6.0, 7.0]])

x_glu = x[..., ::2]
x_glu2 = x[..., 1::2]

x_glu = x_glu2.clamp(min=None, max=7.0)


'''

Input x shape: (N, 2*M) — SwiGLU expects the last dim to be 2 * inner_dim. Typical flow: a Linear projects H -> 2 * inner_dim, you split and apply gated activation; final result has shape (N, inner_dim).
'''

# a = torch.randn( (5, 8), dtype=torch.bfloat16) # (batch_size, seq_length , hidden_size)
# a.dtype
# a.shape
# out = swiglu(a)
# out.shape


# Mixture of Experts
class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        
        self.swiglu_limit = config.swiglu_limit
        
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        '''
        This means:
            If distributed training is initialized (multi-GPU setup):
            → get the total number of participating processes (world_size).

            Else (running on a single GPU or CPU):
            → just use 1.
        '''
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        
        assert config.intermediate_size % self.world_size == 0
        #Ensures intermediate size can be split evenly across distributed ranks if sharding experts.
        
        # Store experts as a list of separate modules to avoid indexing issues
        '''
        With ModuleList, you can:
            add conditionals (e.g. skip some layers)
            reuse layers in special ways (like x = self.layers[i//2](x) + l(x))
            insert dropout, normalization, or attention between layers dynamically
            
            nn.sequential vs nn.modulelist
            ________________________________________________________________________
            - nn.Sequential is a simple way to stack layers, but less flexible.
            self.model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )

            def forward(self, x):
                return self.model(x)   # automatic
            ________________________________________________________________________
            - nn.ModuleList allows for dynamic changes, conditionals, and more complex architectures.
            self.layers = nn.ModuleList([
                nn.Linear(10, 20),
                nn.Linear(20, 5)
            ])

            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))  # manual control
                return x
            _______________________________________________________________________
            
        '''
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(
                    config.hidden_size, 
                    config.intermediate_size * 2 // self.world_size, 
                    device=device, 
                    dtype=torch.bfloat16
                ),
                torch.nn.Linear(
                    config.intermediate_size // self.world_size, 
                    config.hidden_size, 
                    device=device, 
                    dtype=torch.bfloat16
                )
            ) for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_size = x.shape #[5,2880]-->  (seq_length/ tokens-->5, hidden_size-->2880)
        t = self.norm(x) # RMSNorm
        g = self.gate(t) #32 experts logits  #[5,32]--> (seq_length/ tokens-->5, num_experts-->32)
        
        # Get top-k experts
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices
        
        # Flatten for processing
        t_flat = t.view(-1, hidden_size)
        expert_indices_flat = expert_indices.view(-1, self.experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.experts_per_token)
        
        output = torch.zeros_like(t_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            mask = (expert_indices_flat == expert_idx).any(dim=-1)
            
            if not mask.any():
                continue
            
            token_indices = torch.where(mask)[0]
            expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            
            expert_input = t_flat[token_indices]
            weights = expert_weights_flat[token_indices, expert_pos]
            
            # Forward through this expert
            expert_out = expert_input
            expert_out = self.experts[expert_idx][0](expert_out)  # First linear + activation
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            expert_out = self.experts[expert_idx][1](expert_out)  # Second linear
            
            output[token_indices] += expert_out * weights.unsqueeze(-1)
        
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        output = output.view(seq_len, hidden_size)
        return x + output



a = torch.randn( (5, 2880), dtype=torch.bfloat16) # (batch_size, seq_length , hidden_size)
a.dtype
mlp_block = MLPBlock(ModelConfig())
out = mlp_block(a)
out.shape


gate = torch.nn.Linear(
            2880, 32, dtype=torch.bfloat16
        )


out = gate(a)
out.shape



experts = torch.nn.ModuleList([
    torch.nn.Sequential(
        torch.nn.Linear(
            ModelConfig.hidden_size, 
            ModelConfig.intermediate_size * 2,
            dtype=torch.bfloat16
        ),
        torch.nn.Linear(
            ModelConfig.intermediate_size,
            ModelConfig.hidden_size, 
            dtype=torch.bfloat16
        )
    ) for _ in range(ModelConfig.num_experts)
])


'''
    >>> experts
    ModuleList((0-31): 32 x Sequential(
        (0): Linear(in_features=2880, out_features=5760, bias=True)
        (1): Linear(in_features=2880, out_features=2880, bias=True)
    )
    )
'''



class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            # x = self.linears[i // 2](x) + l(x)
            print(f'Layer {i} input shape: ', x.shape)
            x = l(x)
        return x





MyModule().parameters
a = torch.randn( (5, 10)) # (batch_size, seq_length , hidden_size)
model = MyModule()
out = model(a)
out.shape










































































































