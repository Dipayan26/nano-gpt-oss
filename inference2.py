
from architecture.gptoss import Transformer, ModelConfig
import torch
from inference import generate_text

device = "cuda:0"

model = Transformer(ModelConfig(
    num_attention_heads=4,
    num_key_value_heads=4,
    num_experts=4,
    experts_per_token=1,
    num_hidden_layers=4,
    hidden_size=128,
    intermediate_size=128
), device)


context = "Once upon a day"
# These lines should also be inside the block
torch.save(model.state_dict(), r"model\gptoss.pt")
generate_text(model, context)