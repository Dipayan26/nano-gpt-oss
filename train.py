# All your imports go here at the top
from training.data_loader_protein import train_loader, val_loader
# from datasets.dataloader_protein import train_loader, val_loader
from architecture.gptoss import Transformer, ModelConfig
import torch
from inference import generate_text
from training.trainer import trainer




# --- THIS IS THE FIX ---
# Wrap all your "main" code in this block
if __name__ == "__main__":

    device = "cuda:0"
    # context = "Once upon a day"
    
    model = Transformer(ModelConfig(
        num_attention_heads=16,######
        num_key_value_heads=4,
        num_experts=32,
        experts_per_token=4,
        num_hidden_layers=4,
        hidden_size=512,
        intermediate_size=512
    ), device)

    print(sum([p.numel() for p in model.parameters()]) / 1000000, "M parameters")

    # This line will now only be run by the main process
    tl, vl, ts = trainer(model, train_loader, val_loader, device)
    #tl, vl, ts is train_losses,val_losses,tokens_seen is 

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


    # context = "Once upon a day"
    # # These lines should also be inside the block
    # # torch.save(model.state_dict(), r"model\gptoss.pt")
    # generate_text(model, context)
    
    
    
    