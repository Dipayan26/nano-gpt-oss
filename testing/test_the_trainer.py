import torch
from inference import generate_text
import time,os,gc
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from tqdm.notebook import tqdm



def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def calcc(input_batch, target_batch, model,device):
    total_loss=0
    for i in range(len(input_batch)):
        inp = input_batch[i].to(device, non_blocking=True)   
        logits = model(inp)
        del inp
        tgt = target_batch[i].to(device, non_blocking=True)
        loss= torch.nn.functional.cross_entropy(logits,tgt)
        total_loss += loss
        del tgt, logits, loss
        
    clear_gpu_memory()  
    return total_loss/len(input_batch)











