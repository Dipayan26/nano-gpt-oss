import torch
from torch.nn import functional as F

from architecture.tokenizer import get_tokenizer

#_____________________________________________________________
from architecture.gptoss import Transformer, ModelConfig
device = "cuda:0"
# context = "Once upon a day"

model = Transformer(ModelConfig(
    num_attention_heads=4,
    num_key_value_heads=4,
    num_experts=4,
    experts_per_token=1,
    num_hidden_layers=4,
    hidden_size=128,
    intermediate_size=128
), device)

# model.load_state_dict(torch.load(r"model\gotoss_best.pt", weights_only=True))
model.load_state_dict(torch.load(r"model\gptoss.pt", weights_only=True))


# torch.load(model.state_dict(), map_location=r"model\gotoss_best.pt")
#_____________________________________________________________


context_len=8192
tokenizer= get_tokenizer()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())



def generate_text(model, prompt, max_tokens=5, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # Tokenize input
    
    idx = text_to_token_ids(prompt,tokenizer).to(device)
    # Generate
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:] # means we are only conditioning on last context_len tokens which is 8192 will be used for inferance , here in example which is less , but if my promtt is more than 8192 tokens only last 8192 tokens will be used for conditioning why not full context? because of memory constraints
        
        with torch.inference_mode():
            logits= model(idx_cond)
            print(logits.shape)
        logits = logits[-1, :] / temperature
        print(logits.shape)
        print(logits)
        
        # temperature what will do is that if temperature is high like 1.0 or more it will make the distribution more uniform and if temperature is low like 0.5 it will make the distribution more peaky means high probablity tokens will have more probablity and low probablity tokens will have even lower probablity so less randomness in output so optimum is around 0.7 to 0.9
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            print("v is :")
            print(v)
            logits[logits < v[[-1]]] = -float('Inf')
            print(v[[-1]])
            print(logits)
            print(logits.shape)
            

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=0)

    
    # Decode and return
    result = token_ids_to_text(idx,tokenizer)
    return result


context = "paris located in"
# this prompt has 4 tokens 796, 276, 7567, 306 and starts from 0,1,2,3 and will generate 15 more tokens by the model beacause max_tokens=15 by default


generate_text(model, context)


a = torch.randn(4, 4,8)
b= torch.rand(4,201088)

b1 = b[-1,:]# it means last row of b
b1.shape
idx_cond = a[-2:]
idx_cond.shape
min(50, b.size(-1))

z, _ = torch.topk(b, min(50, b.size(-1)))
