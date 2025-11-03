# All your imports go here at the top
from training.data_loader import train_loader, val_loader
from architecture.gptoss import Transformer, ModelConfig
import torch
from inference import generate_text
from training.trainer import trainer

# --- THIS IS THE FIX ---
# Wrap all your "main" code in this block
if __name__ == "__main__":

    device = "cuda:0"
    context = "Once upon a day"
    
    model = Transformer(ModelConfig(
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        experts_per_token=1,
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=256
    ), device)

    print(sum([p.numel() for p in model.parameters()]) / 1000000, "M parameters")

    # This line will now only be run by the main process
    tl, vl, ts = trainer(model, train_loader, val_loader, device)



    # device = "cuda:0"

    # model = Transformer(ModelConfig(
    #     num_attention_heads=4,
    #     num_key_value_heads=4,
    #     num_experts=4,
    #     experts_per_token=1,
    #     num_hidden_layers=4,
    #     hidden_size=128,
    #     intermediate_size=128
    # ), device)


    context = "Once upon a day"
    # These lines should also be inside the block
    torch.save(model.state_dict(), r"model\gptoss.pt")
    generate_text(model, context)