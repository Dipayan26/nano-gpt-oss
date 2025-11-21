import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import torch
import torch.nn as nn

# import wandb
# wandb.login()

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

# help(dataclass)

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
    hidden_size: int = 10
    intermediate_size: int = 10
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
from architecture.tokenizer import get_tokenizer



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
        
        #gate choose the experts for each token, its  a linear layer which will output num_experts logits for each token
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
        t = self.norm(x) # RMSNorm # shape [5,2880]
        g = self.gate(t) #32 experts logits  #[5,32]--> (seq_length/ tokens-->5, num_experts-->32)
        # Get top-k experts means top 4 among 32 experts for each token
        #For each token, take top 4 experts with highest gate score.
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        print(' experts  : ', experts)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        print(' expert weights : ', expert_weights)
        '''
        We only softmax the top-4 scores (not all 32).
        Softmax converts them into probabilities.
        Each row sums to 1.0.
        [2.3, 1.1, 0.5, -0.2] → softmax → [0.63, 0.23, 0.11, 0.03]
        '''
        expert_indices = experts.indices 
        print(' expert indices: ', expert_indices)
        '''
        # means the indices of top 4 experts for each token #This gives you the expert IDs (0 to 31) of those top experts. Shape: [5, 4]  is like for 5 tokens and there is 4 expert indices  for each token. 
        liek this [
                    [7, 3, 12, 1],
                    [4, 19, 22, 7],
                    [7, 3, 12, 1],
                    [4, 19, 22, 7],
                    [4, 19, 22, 7]
                    ]
        '''
        
        # Flatten for processing
        t_flat = t.view(-1, hidden_size) # shape [5, 2880]
        print(' t flat shape: ', t_flat)
        '''
        Today: input is [5, 2880]
        Tomorrow if : input becomes [batch=4, seq=5, 2880]
        Flattening keeps the same MoE logic working.
        '''
        expert_indices_flat = expert_indices.view(-1, self.experts_per_token)#experts_per_token = 4
        print(' expert indices flat : ', expert_indices_flat)
        expert_weights_flat = expert_weights.view(-1, self.experts_per_token)
        print(' expert weights flat : ', expert_weights_flat)
        '''
        t_flat               → [5, 2880]
        expert_indices_flat  → [5, 4]
        expert_weights_flat  → [5, 4]
        '''
        
        output = torch.zeros_like(t_flat)# shape [5, 2880]
        print(' output zerolike : ', output)
        '''
        This creates a zero-initialized tensor of the same shape as t_flat:
        eg - [5, 2880]
        This will hold the final output after processing tokens through their assigned experts.
        ✔ Why is this needed?
        You will soon:
        Send each token → selected experts
        Let each expert process its assigned tokens
        Gather outputs from experts
        Add them back into this output buffer, weighted by routing softmax values
        '''
        
        '''
        Quick reminder of variables before the loop
            seq_len = 5, hidden_size = 2880
            t_flat shape = [5, 2880] (one row per token)
            expert_indices_flat shape = [5, 4] (top-4 expert IDs per token)
            expert_weights_flat shape = [5, 4] (softmax weight per selected expert)
            self.num_experts = 32, self.experts is a list/ModuleList of 32 expert modules (each an MLP, e.g. [Linear, SwiGLU, Linear])
            output shape = [5, 2880] initialized zeros
        '''
        # Process each expert
        for expert_idx in range(self.num_experts):
            print(' Processing expert index: ', expert_idx)
            mask = (expert_indices_flat == expert_idx).any(dim=-1)
            print(' mask for expert index ', expert_idx, ': ', mask)
            '''
            expert_indices_flat == expert_idx returns a boolean tensor of shape [5, 4] telling which of the 4 slots equals expert_idx.
            .any(dim=-1) reduces the 4 slots into a single boolean per token: shape [5].
            mask[i] == True means token i uses this expert_idx in at least one of its top-k slots.
            >>> expert_indices_flat =torch.Tensor([[7, 3,12, 1],
            ...                                     [4,19,22, 7],
            ...                                     [7, 3,12, 1],
            ...                                     [4,19,22, 9],
            ...                                     [4,19,22, 9]]
                If expert_idx = 7: #means which token is using expert 7 
                (==7) =>    [[T,F,F,F],
                            [F,F,F,T],
                            [T,F,F,F],
                            [F,F,F,F],
                            [F,F,F,F]]
                mask => [T, T, T, F, F]   # shape [5]
                for each expert we are getting a list of booleans indicating which tokens are assigned to that expert., the token length here is 5 , but it can be large like If seq_len is large (e.g. 1024, 4096, or more), the mask tensor is size [seq_len] per expert.
                
                if not mask.any():
                continue   ..... means--
                If no token routed to a expert (mask all False) means skip this expert. This avoids unnecessary work.

            '''
            if not mask.any():
                continue
            
            token_indices = torch.where(mask)[0]
            print(' token indices for expert index ', expert_idx, ': ', token_indices)
            '''
            torch.where(mask) returns indices of True entries. [0] extracts the 1-D vector of token indices., 
            so we are getting the indices of tokens that are assigned to this expert(eg - expert 7 (expert_idx)).
            Example:
            eg----
            mask = [True, True, True, False, False]
            token_indices = torch.where(torch.tensor(mask))[0]
            run---
            >>> mask = [True, True, True, False, False]
            >>> token_indices = torch.where(torch.tensor(mask))[0]
            >>> token_indices
            tensor([0, 1, 2])

            '''
            
            expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]# expert position in top-k for each token where it used
            print(' expert positions for expert index ', expert_idx, ': ', expert_pos)
            '''
            so what we are doing we are taking the token_indices whre the token is assigned to this expert and then we are checking in those tokens which position the expert is assigned.
            >>> expert_idx = 7
            >>> token_indices
            tensor([0, 1, 2])
            >>> expert_indices_flat = torch.Tensor([[7, 3,12, 1],
            ...                                     [4,19,22, 7],
            ...                                     [7, 3,12, 1],
            ...                                     [4,19,22, 9],
            ...                                     [4,19,22, 9]]
            ... )
            >>>
            >>> expert_indices_flat[token_indices] == expert_idx
            tensor([[ True, False, False, False],
                    [False, False, False,  True],
                    [ True, False, False, False]])
            >>> (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)
            (tensor([0, 1, 2]), tensor([0, 3, 0]))
            >>> (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            tensor([0, 3, 0]) ## finally getting 
            '''
            
            expert_input = t_flat[token_indices]
            print(' expert input for expert index ', expert_idx, ': ', expert_input)
            '''
            #t_flat shape = [5, 2880] , so we are getting the inputs for the tokens assigned to this expert
            token_indices ==  tensor([0, 1, 2])
            
            expert_input will be of shape [num_tokens_for_this_expert, 2880]
            >>> expert_input = t_flat[token_indices]
            >>> expert_input.shape
            torch.Size([3, 2880])
            #So here expert_input contains the input vectors for only those tokens that are assigned to the current expert (expert_idx).
            '''
            
            
            weights = expert_weights_flat[token_indices, expert_pos]
            print(' weights for expert index ', expert_idx, ': ', weights)
            '''
            weight means how much importance we are giving to this expert for that token( token 7).
            expert_weights_flat[token_indices] =
                [[0.6,0.2,0.1,0.1],   # for token0
                [0.05,0.10,0.20,0.65], # token1
                [0.62,0.18,0.12,0.08]] # token2
                
                expert_pos = [0,3,0] means position of expert 7 
                
                weights = [0.6, 0.65, 0.62]  # shape [3]
            '''
            
            # Forward through this expert
            expert_out = expert_input
            print(' expert out before expert module for expert index ', expert_idx, ': ', expert_out)
            expert_out = self.experts[expert_idx][0](expert_out)  # First linear + activation
            print(' expert out after first linear for expert index ', expert_idx, ': ', expert_out)
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            print(' expert out after swiglu for expert index ', expert_idx, ': ', expert_out)
            expert_out = self.experts[expert_idx][1](expert_out)  # Second linear
            print(' expert out after second linear for expert index ', expert_idx, ': ', expert_out)
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
                    
            previously this separate 35 (num_experts) modules we created where each expert has its own separate two linear layers, now we are using those modules here one by one for each expert. like for expert 7 we are using self.experts[7]
            >>> expert_out = self.experts[expert_idx][0](expert_out)
            here we are using self.experts module from that, choosing module for the expert idx( 7 )  only and then using [0] means first linear layer of that module. the use swiglu and the 2nd linear layer specific to the expert (7).
            and the swiglu make the dimension half so the 1st linear layer putput  is 2*hiddensize then sqiglu make it half then 2nd linear input is normal  hidden size
            numeracally example: 
            >>> expert_out = self.experts[expert_idx][0](expert_out)  # First linear + activation
            >>> expert_out.shape
            torch.Size([3, 5760])  # 2880 * 2 = 5760
            >>> expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            >>> expert_out.shape
            torch.Size([3, 2880])  # back to 2880
            >>> expert_out = self.experts[expert_idx][1](expert_out)  # Second linear
            >>> expert_out.shape
            torch.Size([3, 2880])  # final output shape
            '''
            output[token_indices] += expert_out * weights.unsqueeze(-1)
            
            print(' output after adding weighted expert output for expert index ', expert_idx, ': ', output)
            '''
            ***************************
            in this line if a token passed through expert 2 and also from expert 7 then previous saved outputs from expert 2 will be added with output from the expert 7 for that token.
            So we are accumulating contributions from multiple experts for the same token.
            ***************************
            
            shape  example 
            >>> output[token_indices].shape
            torch.Size([3, 2880])
            >>> weights.unsqueeze(-1).shape  ##weights = [0.6, 0.65, 0.62]  # shape [3]
            torch.Size([3, 1])
            >>> (expert_out * weights.unsqueeze(-1)).shape
            torch.Size([3, 2880])
            eg:
            expert_out =
                [[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]]
            
            weights = [0.6, 0.65, 0.62]
            weights.unsqueeze(-1) is: # expanding the last dim 
                                [[0.6],
                                [0.65],
                                [0.62]]
            
            expert_out * weights.unsqueeze(-1) =
                [[10*0.6, 20*0.6, 30*0.6],
                [40*0.65, 50*0.65, 60*0.65],
                [70*0.62, 80*0.62, 90*0.62]]
                
            final output = [
                            [0.6, 0.6, 0.6],
                            [0.65,0.65,0.65],
                            [0.62,0.62,0.62]
                            ]
            
            output[token_indices] will be updated by adding this final output to it to the corresponding token positions.
            ## created previously 
            output =
                    [
                    [0, 0, 0, ..., 0],   # token 0
                    [0, 0, 0, ..., 0],   # token 1
                    [0, 0, 0, ..., 0],   # token 2
                    [0, 0, 0, ..., 0],   # token 3
                    [0, 0, 0, ..., 0],   # token 4
                    ]
                    
            token_indices = tensor([0, 1, 2])
            
            Meaning:
                token 0 is routed to this expert
                token 1 is routed to this expert
                token 2 is routed to this expert
            
            output[token_indices]+=
            means  
            output[0] += [ 6.00, 12.00, 18.00, 24.00 ]
            output[1] += [32.50, 39.00, 45.50, 52.00]
            output[2] += [ 0.62,  1.24,  1.86,  2.48]
            Other rows (3 and 4) remain zero because those tokens were not routed to this expert.
            so finally
            output =
                        [
                        [ 6.00, 12.00, 18.00, 24.00 ],   # token 0 updated
                        [32.50, 39.00, 45.50, 52.00 ],   # token 1 updated
                        [ 0.62,  1.24,  1.86,  2.48 ],   # token 2 updated
                        [ 0.00,  0.00,  0.00,  0.00 ],   # token 3 untouched
                        [ 0.00,  0.00,  0.00,  0.00 ]    # token 4 untouched
                        ]
            '''
            '''
            This accumulates contributions from multiple experts if a token routes to more than one expert.
            
            here we are adding the expert output to the final output buffer for the tokens assigned to this expert.
            weights.unsqueeze(-1) changes shape from [num_tokens_for_this_expert] to [num_tokens_for_this_expert, 1] so it can broadcast during multiplication.
            This scales each token’s expert output by the routing softmax weight before adding it to the final output.
            
            
            ***************************
            in this line if a token passed through expert 2 and also from expert 7 then previous saved outputs from expert 2 will be added with output from the expert 7 for that token.
            So we are accumulating contributions from multiple experts for the same token.
            ***************************
            '''
        
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        output = output.view(seq_len, hidden_size)
        print(' output after all_reduce and view: ', output)
        return x + output
        # return output



# a = torch.randn( (5, 10), dtype=torch.bfloat16) # (batch_size, seq_length , hidden_size)

a = torch.tensor([[0.5, 1.0, 2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0],
                  [4.0, 10.0, 6.0, 7.0,8.0,9.0,10.0,11.0,12.0,13.0],
                  [1.0, 2.0, 3.0, 4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                  [5.0, 11.0, 7.0, 8.0,9.0,10.0,11.0,12.0,13.0,14.0],
                  [6.0, 12.0, 8.0, 9.0,10.0,11.0,12.0,13.0,14.0,15.0]], dtype=torch.bfloat16)

a.dtype
mlp_block = MLPBlock(ModelConfig())
out = mlp_block(a)
# out.shape

'''
@dataclass
class ModelConfig:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 10 ********************* usually 2880
    intermediate_size: int = 10 *********************** usually 2880
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

# a = torch.randn( (5, 10), dtype=torch.bfloat16) # (batch_size, seq_length , hidden_size)

a = torch.tensor([[0.5, 1.0, 2.0, 3.0,4.0,5.0,6.0,7.0,8.0,9.0],
                  [4.0, 10.0, 6.0, 7.0,8.0,9.0,10.0,11.0,12.0,13.0],
                  [1.0, 2.0, 3.0, 4.0,5.0,6.0,7.0,8.0,9.0,10.0],
                  [5.0, 11.0, 7.0, 8.0,9.0,10.0,11.0,12.0,13.0,14.0],
                  [6.0, 12.0, 8.0, 9.0,10.0,11.0,12.0,13.0,14.0,15.0]], dtype=torch.bfloat16)

a.dtype
mlp_block = MLPBlock(ModelConfig())
out = mlp_block(a)
# out.shape

>>> out = mlp_block(a)

 experts:  torch.return_types.topk( **********************************
values=tensor([[1.0703, 0.9844, 0.8555, 0.7969],
        [0.9766, 0.8242, 0.7773, 0.6875],
        [0.9961, 0.9336, 0.9062, 0.7656],
        [1.0000, 0.8281, 0.7422, 0.6836],
        [1.0156, 0.8281, 0.7148, 0.6797]], dtype=torch.bfloat16,
       grad_fn=<TopkBackward0>),
indices=tensor([[17, 25,  3,  5],
        [ 3, 21, 25, 13],
        [17, 25,  3,  5],
        [ 3, 21, 25, 13],
        [ 3, 21, 25, 13]]))

 expert weights :  tensor([[0.2871, 0.2637, 0.2314, 0.2188],************************
        [0.2910, 0.2500, 0.2393, 0.2188],
        [0.2734, 0.2578, 0.2500, 0.2178],
        [0.2988, 0.2520, 0.2314, 0.2178],
        [0.3047, 0.2520, 0.2256, 0.2178]], dtype=torch.bfloat16,
       grad_fn=<SoftmaxBackward0>)
       
 expert indices:  tensor([[17, 25,  3,  5],
        [ 3, 21, 25, 13],
        [17, 25,  3,  5],
        [ 3, 21, 25, 13],
        [ 3, 21, 25, 13]])
        
        
 t flat shape:  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<ViewBackward0>)
 expert indices flat :  tensor([[17, 25,  3,  5],
        [ 3, 21, 25, 13],
        [17, 25,  3,  5],
        [ 3, 21, 25, 13],
        [ 3, 21, 25, 13]])
 expert weights flat :  tensor([[0.2871, 0.2637, 0.2314, 0.2188],
        [0.2910, 0.2500, 0.2393, 0.2188],
        [0.2734, 0.2578, 0.2500, 0.2178],
        [0.2988, 0.2520, 0.2314, 0.2178],
        [0.3047, 0.2520, 0.2256, 0.2178]], dtype=torch.bfloat16,
       grad_fn=<ViewBackward0>)
 output zerolike :  tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.bfloat16)

*****************************************************
 Processing expert index:  0
 mask for expert index  0 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  1
 mask for expert index  1 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  2
 mask for expert index  2 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  3
 mask for expert index  3 :  tensor([True, True, True, True, True])
 token indices for expert index  3 :  tensor([0, 1, 2, 3, 4])
 expert positions for expert index  3 :  tensor([2, 0, 2, 0, 0])
 expert input for expert index  3 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 weights for expert index  3 :  tensor([0.2314, 0.2910, 0.2500, 0.2988, 0.3047], dtype=torch.bfloat16,
       grad_fn=<IndexBackward0>)
 expert out before expert module for expert index  3 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 expert out after first linear for expert index  3 :  tensor([[-0.9961,  0.0203, -0.4512,  0.1797, -0.2578,  0.3652, -1.1875, -0.5273,
          0.0938,  0.8125,  0.5898, -0.3203,  0.5352,  0.2715,  0.6445,  0.0854,
         -0.9414, -1.0938, -0.4902, -0.4805],
        [-0.7266,  0.2363, -0.4688,  0.1885, -0.3418,  0.5586, -1.0078, -0.6016,
         -0.3594,  0.8672,  0.4727, -0.1553,  0.4492,  0.2832,  0.6523,  0.5586,
         -0.9766, -1.0781, -0.5117, -0.9297],
        [-0.9727,  0.0544, -0.4531,  0.1738, -0.2852,  0.4102, -1.1172, -0.5430,
         -0.0146,  0.8242,  0.5742, -0.2773,  0.5508,  0.2793,  0.6133,  0.1924,
         -0.9648, -1.1250, -0.4707, -0.5938],
        [-0.7344,  0.2520, -0.4648,  0.1729, -0.3457,  0.5586, -0.9727, -0.6016,
         -0.3926,  0.8672,  0.4551, -0.1572,  0.4688,  0.2773,  0.6211,  0.5938,
         -0.9883, -1.0781, -0.5000, -0.9648],
        [-0.7383,  0.2656, -0.4629,  0.1611, -0.3477,  0.5547, -0.9453, -0.6016,
         -0.4258,  0.8672,  0.4434, -0.1611,  0.4844,  0.2734,  0.5977,  0.6250,
         -0.9961, -1.0859, -0.4922, -0.9961]], dtype=torch.bfloat16,
       grad_fn=<AddmmBackward0>)
 expert out after swiglu for expert index  3 :  tensor([[-1.5820e-01, -1.6797e-01, -1.3867e-01, -6.5918e-02,  9.1797e-02,
          2.9492e-01,  4.8828e-01,  5.2734e-01,  1.4832e-02, -7.7148e-02],
        [-2.0215e-01, -1.7285e-01, -1.9238e-01, -6.1035e-02, -2.3535e-01,
          2.7539e-01,  3.9258e-01,  7.6953e-01,  1.2146e-02, -1.0620e-02],
        [-1.6504e-01, -1.6797e-01, -1.5234e-01, -6.6406e-02, -1.3184e-02,
          3.0273e-01,  5.0781e-01,  5.4297e-01,  1.9531e-02, -5.9570e-02],
        [-2.0410e-01, -1.7090e-01, -1.9336e-01, -6.2256e-02, -2.5000e-01,
          2.6172e-01,  4.1211e-01,  7.3438e-01,  1.2146e-02, -5.2490e-03],
        [-2.0801e-01, -1.6797e-01, -1.9238e-01, -6.2988e-02, -2.5977e-01,
          2.5195e-01,  4.2773e-01,  7.1484e-01,  1.3245e-02, -5.8365e-04]],
       dtype=torch.bfloat16, grad_fn=<MulBackward0>)
 expert out after second linear for expert index  3 :  tensor([[ 0.0991, -0.0581,  0.2422,  0.0596,  0.2168, -0.2539, -0.2021,  0.1416,
          0.1631,  0.1152],
        [ 0.2734, -0.0603,  0.1973,  0.0986,  0.1992, -0.3418, -0.0732,  0.1738,
          0.1729,  0.1533],
        [ 0.1416, -0.0815,  0.2197,  0.0728,  0.2002, -0.2832, -0.1699,  0.1514,
          0.1729,  0.1416],
        [ 0.2695, -0.0776,  0.1904,  0.1064,  0.1963, -0.3418, -0.0732,  0.1777,
          0.1768,  0.1650],
        [ 0.2676, -0.0874,  0.1855,  0.1099,  0.1953, -0.3457, -0.0728,  0.1816,
          0.1787,  0.1729]], dtype=torch.bfloat16, grad_fn=<AddmmBackward0>)
 output after adding weighted expert output for expert index  3 :  tensor([[ 0.0229, -0.0134,  0.0562,  0.0138,  0.0503, -0.0588, -0.0469,  0.0327,
          0.0378,  0.0266],
        [ 0.0796, -0.0176,  0.0574,  0.0287,  0.0579, -0.0996, -0.0214,  0.0505,
          0.0503,  0.0447],
        [ 0.0354, -0.0204,  0.0549,  0.0182,  0.0500, -0.0708, -0.0425,  0.0378,
          0.0432,  0.0354],
        [ 0.0806, -0.0232,  0.0569,  0.0317,  0.0586, -0.1021, -0.0219,  0.0532,
          0.0527,  0.0493],
        [ 0.0815, -0.0266,  0.0566,  0.0334,  0.0596, -0.1055, -0.0222,  0.0554,
          0.0544,  0.0527]], dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)
*****************************************************
 Processing expert index:  4
 mask for expert index  4 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  5
 mask for expert index  5 :  tensor([ True, False,  True, False, False])
 token indices for expert index  5 :  tensor([0, 2])
 expert positions for expert index  5 :  tensor([3, 3])
 expert input for expert index  5 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 weights for expert index  5 :  tensor([0.2188, 0.2178], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)        
 expert out before expert module for expert index  5 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 expert out after first linear for expert index  5 :  tensor([[ 0.1289,  0.0923, -1.0547,  0.5469, -0.3008,  0.5430, -0.4492, -0.3086,
          0.8828, -0.1406,  0.3438,  0.1250,  0.8633, -1.0156,  0.0757,  0.0613,
         -0.4766,  0.3340,  0.3066, -0.7852],
        [ 0.1768,  0.0237, -1.1250,  0.5586, -0.2832,  0.5000, -0.3613, -0.2324,
          0.8945, -0.2061,  0.3887,  0.1279,  0.8438, -0.9570, -0.0031,  0.0413,
         -0.5391,  0.4102,  0.2852, -0.6875]], dtype=torch.bfloat16,
       grad_fn=<AddmmBackward0>)
 expert out after swiglu for expert index  5 :  tensor([[ 0.0781, -0.2324, -0.1748, -0.0986,  0.6211,  0.2480, -0.0110,  0.0427,
         -0.1953,  0.0413],
        [ 0.1040, -0.2256, -0.1621, -0.0972,  0.5820,  0.2871,  0.0294, -0.0016,
         -0.2158,  0.0549]], dtype=torch.bfloat16, grad_fn=<MulBackward0>)
 expert out after second linear for expert index  5 :  tensor([[-0.2080, -0.3301, -0.3828,  0.3262,  0.0435,  0.1660, -0.0500, -0.1338,
          0.1973, -0.3574],
        [-0.1875, -0.3145, -0.3730,  0.3301,  0.0447,  0.1846, -0.0605, -0.1533,
          0.2051, -0.3457]], dtype=torch.bfloat16, grad_fn=<AddmmBackward0>)
 output after adding weighted expert output for expert index  5 :  tensor([[-0.0225, -0.0859, -0.0278,  0.0850,  0.0598, -0.0225, -0.0579,  0.0034,
          0.0811, -0.0515],
        [ 0.0796, -0.0176,  0.0574,  0.0287,  0.0579, -0.0996, -0.0214,  0.0505,
          0.0503,  0.0447],
        [-0.0054, -0.0889, -0.0261,  0.0898,  0.0598, -0.0305, -0.0557,  0.0044,
          0.0879, -0.0398],
        [ 0.0806, -0.0232,  0.0569,  0.0317,  0.0586, -0.1021, -0.0219,  0.0532,
          0.0527,  0.0493],
        [ 0.0815, -0.0266,  0.0566,  0.0334,  0.0596, -0.1055, -0.0222,  0.0554,
          0.0544,  0.0527]], dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)
*****************************************************
 Processing expert index:  6
 mask for expert index  6 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  7
 mask for expert index  7 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  8
 mask for expert index  8 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  9
 mask for expert index  9 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  10
 mask for expert index  10 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  11
 mask for expert index  11 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  12
 mask for expert index  12 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  13
 mask for expert index  13 :  tensor([False,  True, False,  True,  True])
 token indices for expert index  13 :  tensor([1, 3, 4])
 expert positions for expert index  13 :  tensor([3, 3, 3])************
 expert input for expert index  13 :  tensor([[0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 weights for expert index  13 :  tensor([0.2188, 0.2178, 0.2178], dtype=torch.bfloat16,
       grad_fn=<IndexBackward0>)
 expert out before expert module for expert index  13 :  tensor([[0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 expert out after first linear for expert index  13 :  tensor([[-0.1953, -0.1543, -1.3438, -0.3223, -0.7344,  0.4570, -0.2217,  0.7031,
          0.2910, -0.0757, -0.4023,  1.1953,  0.3320, -0.3477,  0.3906,  0.4199,
         -0.0664,  0.5156, -0.8984,  0.5664],
        [-0.1904, -0.1650, -1.3594, -0.3457, -0.7227,  0.4629, -0.2090,  0.6953,
          0.3047, -0.1045, -0.3848,  1.2031,  0.3418, -0.3320,  0.3555,  0.4121,
         -0.0762,  0.5352, -0.9102,  0.5820],
        [-0.1875, -0.1729, -1.3672, -0.3633, -0.7148,  0.4688, -0.2021,  0.6914,
          0.3145, -0.1289, -0.3750,  1.2188,  0.3477, -0.3223,  0.3301,  0.4082,
         -0.0864,  0.5547, -0.9219,  0.5938]], dtype=torch.bfloat16,
       grad_fn=<AddmmBackward0>)
 expert out after swiglu for expert index  13 :  tensor([[-0.0688, -0.0845, -0.2373, -0.1533,  0.1670, -0.2949,  0.1377,  0.3672,
         -0.0479, -0.2500],
        [-0.0669, -0.0806, -0.2393, -0.1455,  0.1709, -0.2910,  0.1465,  0.3262,
         -0.0547, -0.2520],
        [-0.0654, -0.0776, -0.2393, -0.1406,  0.1729, -0.2891,  0.1523,  0.2949,
         -0.0623, -0.2520]], dtype=torch.bfloat16, grad_fn=<MulBackward0>)
 expert out after second linear for expert index  13 :  tensor([[ 0.1167,  0.2812, -0.0894, -0.2285, -0.2090,  0.0457, -0.2197, -0.1572,
          0.1387,  0.1055],
        [ 0.1196,  0.2793, -0.0796, -0.2266, -0.1943,  0.0308, -0.2119, -0.1631,
          0.1309,  0.0952],
        [ 0.1221,  0.2793, -0.0718, -0.2256, -0.1846,  0.0198, -0.2041, -0.1670,
          0.1245,  0.0874]], dtype=torch.bfloat16, grad_fn=<AddmmBackward0>)
 output after adding weighted expert output for expert index  13 :  tensor([[-0.0225, -0.0859, -0.0278,  0.0850,  0.0598, -0.0225, -0.0579,  0.0034,
          0.0811, -0.0515],
        [ 0.1050,  0.0439,  0.0378, -0.0214,  0.0122, -0.0898, -0.0693,  0.0161,
          0.0806,  0.0679],
        [-0.0054, -0.0889, -0.0261,  0.0898,  0.0598, -0.0305, -0.0557,  0.0044,
          0.0879, -0.0398],
        [ 0.1064,  0.0376,  0.0396, -0.0176,  0.0164, -0.0952, -0.0679,  0.0178,
          0.0811,  0.0703],
        [ 0.1084,  0.0342,  0.0410, -0.0156,  0.0193, -0.1011, -0.0664,  0.0190,
          0.0815,  0.0718]], dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)
*****************************************************
 Processing expert index:  14
 mask for expert index  14 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  15
 mask for expert index  15 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  16
 mask for expert index  16 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  17
 mask for expert index  17 :  tensor([ True, False,  True, False, False])
 token indices for expert index  17 :  tensor([0, 2])
 expert positions for expert index  17 :  tensor([0, 0])
 expert input for expert index  17 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 weights for expert index  17 :  tensor([0.2871, 0.2734], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)       
 expert out before expert module for expert index  17 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 expert out after first linear for expert index  17 :  tensor([[-1.0469,  0.9570,  1.1484, -0.1787, -0.1069, -0.4043, -0.3379,  0.3125,
         -0.0530, -0.0016,  0.1777,  0.2012,  0.5820, -0.0559,  0.3418, -0.4941,
         -1.1328,  0.2471, -0.4199, -1.2188],
        [-0.9961,  0.8789,  1.1719, -0.1328, -0.0208, -0.4121, -0.4062,  0.3926,
         -0.0074,  0.0165,  0.1719,  0.1338,  0.5508, -0.0894,  0.2734, -0.4980,
         -1.1250,  0.2773, -0.4824, -1.2969]], dtype=torch.bfloat16,
       grad_fn=<AddmmBackward0>)
 expert out after swiglu for expert index  17 :  tensor([[-0.2949,  0.8281, -0.0288, -0.1592, -0.0253,  0.1226,  0.4023,  0.1113,
         -0.1797,  0.0302],
        [-0.2891,  0.8945, -0.0060, -0.1885, -0.0037,  0.1118,  0.3613,  0.0840,
         -0.1855,  0.0437]], dtype=torch.bfloat16, grad_fn=<MulBackward0>)
 expert out after second linear for expert index  17 :  tensor([[ 0.4805,  0.0889, -0.1807, -0.3398,  0.1807,  0.2520, -0.0386, -0.2373,
         -0.2178,  0.3047],
        [ 0.4980,  0.1006, -0.1777, -0.3262,  0.1904,  0.2676, -0.0410, -0.2617,
         -0.2227,  0.3398]], dtype=torch.bfloat16, grad_fn=<AddmmBackward0>)
 output after adding weighted expert output for expert index  17 :  tensor([[ 0.1152, -0.0605, -0.0796, -0.0127,  0.1113,  0.0498, -0.0688, -0.0649,
          0.0186,  0.0359],
        [ 0.1050,  0.0439,  0.0378, -0.0214,  0.0122, -0.0898, -0.0693,  0.0161,
          0.0806,  0.0679],
        [ 0.1309, -0.0615, -0.0747,  0.0005,  0.1118,  0.0427, -0.0669, -0.0674,
          0.0271,  0.0530],
        [ 0.1064,  0.0376,  0.0396, -0.0176,  0.0164, -0.0952, -0.0679,  0.0178,
          0.0811,  0.0703],
        [ 0.1084,  0.0342,  0.0410, -0.0156,  0.0193, -0.1011, -0.0664,  0.0190,
          0.0815,  0.0718]], dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)
*****************************************************
 Processing expert index:  18
 mask for expert index  18 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  19
 mask for expert index  19 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  20
 mask for expert index  20 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  21
 mask for expert index  21 :  tensor([False,  True, False,  True,  True])
 token indices for expert index  21 :  tensor([1, 3, 4])
 expert positions for expert index  21 :  tensor([1, 1, 1])
 expert input for expert index  21 :  tensor([[0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 weights for expert index  21 :  tensor([0.2500, 0.2520, 0.2520], dtype=torch.bfloat16,
       grad_fn=<IndexBackward0>)
 expert out before expert module for expert index  21 :  tensor([[0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 expert out after first linear for expert index  21 :  tensor([[-0.5000, -0.7773, -0.0698, -0.2344, -0.7266, -0.2109, -0.0645,  0.4434,
         -1.1406, -0.1309, -1.0469, -0.6328, -0.1689,  0.1797,  1.1562, -0.1387,
          0.1553,  0.1206,  0.2207,  1.1406],
        [-0.4766, -0.7617, -0.0449, -0.2412, -0.7344, -0.2305, -0.0718,  0.4512,
         -1.1406, -0.1118, -1.0469, -0.6445, -0.1758,  0.1709,  1.1406, -0.1328,
          0.1562,  0.1270,  0.2080,  1.1562],
        [-0.4551, -0.7539, -0.0220, -0.2461, -0.7383, -0.2500, -0.0771,  0.4590,
         -1.1406, -0.0957, -1.0391, -0.6523, -0.1807,  0.1621,  1.1328, -0.1318,
          0.1572,  0.1318,  0.2002,  1.1719]], dtype=torch.bfloat16,
       grad_fn=<AddmmBackward0>)
 expert out after swiglu for expert index  21 :  tensor([[-0.0332, -0.0253, -0.1299, -0.0442, -0.1245, -0.0557, -0.0859,  0.8711,
          0.0981,  0.2793],
        [-0.0349, -0.0164, -0.1260, -0.0491, -0.1270, -0.0537, -0.0874,  0.8672,
          0.0996,  0.2637],
        [-0.0354, -0.0081, -0.1230, -0.0527, -0.1299, -0.0530, -0.0894,  0.8594,
          0.1006,  0.2539]], dtype=torch.bfloat16, grad_fn=<MulBackward0>)
 expert out after second linear for expert index  21 :  tensor([[ 0.1455,  0.3633,  0.3438, -0.2197,  0.0437,  0.4121,  0.1021,  0.1118,
         -0.4961, -0.0898],
        [ 0.1406,  0.3594,  0.3398, -0.2197,  0.0442,  0.4102,  0.0991,  0.1040,
         -0.4961, -0.0947],
        [ 0.1396,  0.3574,  0.3359, -0.2197,  0.0461,  0.4082,  0.0986,  0.0977,
         -0.4941, -0.0977]], dtype=torch.bfloat16, grad_fn=<AddmmBackward0>)
 output after adding weighted expert output for expert index  21 :  tensor([[ 0.1152, -0.0605, -0.0796, -0.0127,  0.1113,  0.0498, -0.0688, -0.0649,
          0.0186,  0.0359],
        [ 0.1416,  0.1348,  0.1240, -0.0762,  0.0232,  0.0132, -0.0439,  0.0439,
         -0.0435,  0.0454],
        [ 0.1309, -0.0615, -0.0747,  0.0005,  0.1118,  0.0427, -0.0669, -0.0674,
          0.0271,  0.0530],
        [ 0.1416,  0.1279,  0.1250, -0.0732,  0.0275,  0.0083, -0.0430,  0.0439,
         -0.0439,  0.0464],
        [ 0.1436,  0.1240,  0.1250, -0.0713,  0.0309,  0.0020, -0.0415,  0.0437,
         -0.0430,  0.0471]], dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)
*****************************************************
 Processing expert index:  22
 mask for expert index  22 :  tensor([False, False, False, False, False])
 Processing expert index:  23
 mask for expert index  23 :  tensor([False, False, False, False, False])
 Processing expert index:  24
 mask for expert index  24 :  tensor([False, False, False, False, False])
*****************************************************
 Processing expert index:  25
 mask for expert index  25 :  tensor([True, True, True, True, True])
 token indices for expert index  25 :  tensor([0, 1, 2, 3, 4])
 expert positions for expert index  25 :  tensor([1, 2, 1, 2, 2])
 expert input for expert index  25 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 weights for expert index  25 :  tensor([0.2637, 0.2393, 0.2578, 0.2314, 0.2256], dtype=torch.bfloat16,
       grad_fn=<IndexBackward0>)
 expert out before expert module for expert index  25 :  tensor([[0.0938, 0.1875, 0.3750, 0.5625, 0.7500, 0.9375, 1.1250, 1.3125, 1.5000,
         1.6875],
        [0.4258, 1.0625, 0.6406, 0.7461, 0.8516, 0.9609, 1.0625, 1.1719, 1.2812,
         1.3828],
        [0.1611, 0.3223, 0.4844, 0.6445, 0.8047, 0.9688, 1.1250, 1.2891, 1.4531,
         1.6094],
        [0.4824, 1.0625, 0.6758, 0.7734, 0.8711, 0.9648, 1.0625, 1.1562, 1.2578,
         1.3516],
        [0.5312, 1.0625, 0.7070, 0.7969, 0.8828, 0.9727, 1.0625, 1.1484, 1.2344,
         1.3281]], dtype=torch.bfloat16, grad_fn=<IndexBackward0>)
 expert out after first linear for expert index  25 :  tensor([[-0.3438, -0.2969,  0.8477,  0.0742, -0.2500, -0.6680,  0.2891, -0.9961,
          0.5117,  0.3867, -0.3613, -0.7148,  0.9805, -0.5078, -0.0757,  0.1582,
         -0.9531, -0.5039,  0.7266,  0.1641],
        [-0.4590, -0.2891,  0.6680, -0.0344, -0.0052, -0.7578,  0.2852, -0.7188,
          0.4238,  0.6016, -0.5273, -0.5898,  0.9297, -0.7891, -0.1318,  0.4219,
         -0.5156, -0.8984,  0.6914,  0.3965],
          0.4160,  0.6172, -0.5273, -0.5898,  0.9141, -0.8047, -0.1206,  0.4199,
         -0.4766, -0.9258,  0.6836,  0.4121],
          0.4160,  0.6172, -0.5273, -0.5898,  0.9141, -0.8047, -0.1206,  0.4199,
         -0.4766, -0.9258,  0.6836,  0.4121],
        [-0.5117, -0.3008,  0.6250, -0.0854,  0.0417, -0.8008,  0.2812, -0.6680,
          0.4102,  0.6328, -0.5273, -0.5938,  0.9062, -0.8164, -0.1118,  0.4199,
         -0.4453, -0.9531,  0.6797,  0.4258]], dtype=torch.bfloat16,
       grad_fn=<AddmmBackward0>)
 expert out after swiglu for expert index  25 :  tensor([[-8.6426e-02,  7.3828e-01, -3.2715e-02,  7.0190e-04,  5.0000e-01,    
         -3.6133e-02,  4.0820e-01, -4.1016e-02, -7.7637e-02,  6.5625e-01],
        [-1.0254e-01,  4.9023e-01, -6.2180e-04,  4.9316e-02,  4.5703e-01,
         -6.2500e-02,  1.6211e-01, -8.3008e-02, -1.5381e-02,  7.4219e-01],
        [-9.7168e-02,  6.6016e-01, -2.2461e-02,  1.0742e-02,  4.7070e-01,
         -4.1260e-02,  3.4375e-01, -4.3945e-02, -6.5430e-02,  6.9141e-01],
        [-1.0449e-01,  4.5312e-01,  2.2583e-03,  5.3711e-02,  4.4922e-01,
         -6.2500e-02,  1.4746e-01, -7.7148e-02, -1.0864e-02,  7.3438e-01],
        [-1.0596e-01,  4.2578e-01,  4.3335e-03,  5.7617e-02,  4.4727e-01,
         -6.2012e-02,  1.3672e-01, -7.2266e-02, -6.6528e-03,  7.3828e-01]],
       dtype=torch.bfloat16, grad_fn=<MulBackward0>)
 expert out after second linear for expert index  25 :  tensor([[-0.1777,  0.1689,  0.0942,  0.2178, -0.2109, -0.0093,  0.1226, -0.0806,
         -0.2090,  0.2402],
        [-0.2393,  0.1465,  0.0289,  0.1030, -0.3086, -0.0610,  0.1357, -0.1924,
         -0.1826,  0.2109],
        [-0.1953,  0.1650,  0.0718,  0.1836, -0.2383, -0.0199,  0.1230, -0.1069,
         -0.2021,  0.2314],
        [-0.2383,  0.1416,  0.0231,  0.1006, -0.3125, -0.0564,  0.1387, -0.1982,
         -0.1738,  0.2021],
        [-0.2412,  0.1377,  0.0197,  0.0962, -0.3184, -0.0554,  0.1406, -0.2051,
         -0.1689,  0.1992]], dtype=torch.bfloat16, grad_fn=<AddmmBackward0>)
 output after adding weighted expert output for expert index  25 :  tensor([[ 0.0684, -0.0161, -0.0547,  0.0447,  0.0557,  0.0474, -0.0366, -0.0859,
         -0.0366,  0.0996],
        [ 0.0845,  0.1699,  0.1309, -0.0515, -0.0505, -0.0014, -0.0115, -0.0022,
         -0.0869,  0.0957],
        [ 0.0806, -0.0190, -0.0562,  0.0479,  0.0503,  0.0376, -0.0352, -0.0947,
         -0.0249,  0.1123],
        [ 0.0864,  0.1602,  0.1299, -0.0498, -0.0449, -0.0048, -0.0110, -0.0020,
         -0.0840,  0.0933],
        [ 0.0889,  0.1553,  0.1299, -0.0496, -0.0410, -0.0106, -0.0098, -0.0024,
         -0.0811,  0.0918]], dtype=torch.bfloat16, grad_fn=<IndexPutBackward0>)
*****************************************************
 Processing expert index:  26
 mask for expert index  26 :  tensor([False, False, False, False, False])
 Processing expert index:  27
 mask for expert index  27 :  tensor([False, False, False, False, False])
 Processing expert index:  28
 mask for expert index  28 :  tensor([False, False, False, False, False])
 Processing expert index:  29
 mask for expert index  29 :  tensor([False, False, False, False, False])
 Processing expert index:  30
 mask for expert index  30 :  tensor([False, False, False, False, False])
 Processing expert index:  31
 mask for expert index  31 :  tensor([False, False, False, False, False])
 
*****************************************************
*****************************************************
*****************************************************
 output after all_reduce and view:  tensor([[ 0.0684, -0.0161, -0.0547,  0.0447,  0.0557,  0.0474, -0.0366, -0.0859,
         -0.0366,  0.0996],
        [ 0.0845,  0.1699,  0.1309, -0.0515, -0.0505, -0.0014, -0.0115, -0.0022,
         -0.0869,  0.0957],
        [ 0.0806, -0.0190, -0.0562,  0.0479,  0.0503,  0.0376, -0.0352, -0.0947,
         -0.0249,  0.1123],
        [ 0.0864,  0.1602,  0.1299, -0.0498, -0.0449, -0.0048, -0.0110, -0.0020,
         -0.0840,  0.0933],
        [ 0.0889,  0.1553,  0.1299, -0.0496, -0.0410, -0.0106, -0.0098, -0.0024,
         -0.0811,  0.0918]], dtype=torch.bfloat16, grad_fn=<ViewBackward0>)
'''








my_dict = {
    "key1": ["valueA", "valueB", "valueC"],
    "key2": ["valueX", "valueY"]
}

# Adding a new value to an existing key
my_dict["key1"].append("valueD")

# Accessing values
print(my_dict["key1"])  # Output: ['valueA', 'valueB', 'valueC', 'valueD']
print(my_dict["key1"][0]) # Output: valueA


my_dict = {"name": "Alice", "age": 30}
my_dict["city"] = "New York"
print(my_dict)
# Output: {'name': 'Alice', 'age': 30, 'city': 'New York'}







a = [ 18049,   7557,    261,   2163, 181110, 126431, 174852,   1386,  98232,
        150381,  50886,  94929, 127426,  99477,  67433, 175191,  53520, 149700,
        137793,  56558, 126140, 174320, 105012, 160121, 106162,  64111, 130982,
        165840,  73471, 181884, 143700, 152113,  20144,  30311, 182616, 104182,
        110371,  56988, 126196,  76229,  39346, 118494, 133216, 114313,  32690,
         90497,  40216,  22723, 165456, 118388, 165505, 161172, 107882, 125632,
        165268, 140893,  30559, 196652, 195459,  16093, 125049,  28480,  46828,
        130050, 152707,  90721,  56931, 142329, 166089, 157585,   1102, 145466,
         70332, 188677, 126739,  49636, 187936,  90735, 123117, 189602,  17310,
        141168,  57809,  69815, 199190, 136918,  50036, 123810,  55430, 190382,
        199596, 116724,   1412,   1058,  71272,  67168, 148436, 190276, 128814,
          7472, 110425, 195330,  69860]


len(a)





'''
rinting Input : tensor([ 1770,  2023, 63202,   261,  5057,   402,  1335, 27628,   364,    43],
       device='cuda:0')
 experts  :  torch.Size([10, 4])
layer idx : 0
 experts  :  torch.Size([10, 4])
layer idx : 1
 experts  :  torch.Size([10, 4])
layer idx : 2
 experts  :  torch.Size([10, 4])
layer idx : 3
 experts  :  torch.Size([10, 4])
layer idx : 4
 experts  :  torch.Size([10, 4])
layer idx : 5
 experts  :  torch.Size([10, 4])
layer idx : 6
 experts  :  torch.Size([10, 4])
layer idx : 7
printing Input : tensor([  5143,    290,  51196,    326,   9295,    634,  27628,   6635, 108008,
            11], device='cuda:0')
 experts  :  torch.Size([10, 4])
layer idx : 0
 experts  :  torch.Size([10, 4])
layer idx : 1
 experts  :  torch.Size([10, 4])
layer idx : 2
 experts  :  torch.Size([10, 4])
layer idx : 3
 experts  :  torch.Size([10, 4])
layer idx : 4
 experts  :  torch.Size([10, 4])
layer idx : 5
 experts  :  torch.Size([10, 4])
layer idx : 6
 experts  :  torch.Size([10, 4])
layer idx : 7
printing Input : tensor([ 7201,   316,  5143,   290, 51196,   483,  1335,  3317,    11,   813],
       device='cuda:0')
 experts  :  torch.Size([10, 4])
layer idx : 0
 experts  :  torch.Size([10, 4])
layer idx : 1
 experts  :  torch.Size([10, 4])
layer idx : 2
 experts  :  torch.Size([10, 4])
layer idx : 3
 experts  :  torch.Size([10, 4])
layer idx : 4
 experts  :  torch.Size([10, 4])
layer idx : 5
 experts  :  torch.Size([10, 4])
layer idx : 6
 experts  :  torch.Size([10, 4])
layer idx : 7

'''


















































