import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

import torch
import torch.nn as nn

vocab_size = 50000
embed_dim = 10

embedding_layer = nn.Embedding(vocab_size, embed_dim)
print(embedding_layer.weight.shape)  # torch.Size([50000, 768])


embedding_layer.weight[0]  # shape (768,)
embedding_layer.weight[1]  # shape (768,)
embedding_layer.weight[2]  # shape (768,)

embedding_layer.weight[49000]  # shape (768,)
embedding_layer.weight[50000]  # shape (768,)
















