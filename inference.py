import torch
from torch.nn import functional as F

from architecture.tokenizer import get_tokenizer



# #_____________________________________________________________
# from architecture.gptoss import Transformer, ModelConfig
# device = "cuda:0"
# # context = "Once upon a day"

# model = Transformer(ModelConfig(
#     num_attention_heads=4,
#     num_key_value_heads=4,
#     num_experts=4,
#     experts_per_token=1,
#     num_hidden_layers=4,
#     hidden_size=128,
#     intermediate_size=128
# ), device)

# # model.load_state_dict(torch.load(r"model\gotoss_best.pt", weights_only=True))
# model.load_state_dict(torch.load(r"model\gptoss.pt", weights_only=True))


# # torch.load(model.state_dict(), map_location=r"model\gotoss_best.pt")
# #_____________________________________________________________



context_len=8192
tokenizer= get_tokenizer()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())



def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # Tokenize input
    
    idx = text_to_token_ids(prompt,tokenizer).to(device)
    # Generate
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:] 
        '''
        # means we are only conditioning on last context_len tokens which is 8192 will be used for inferance , here in example which is less , but if my promtt is more than 8192 tokens only last 8192 tokens will be used for conditioning why not full context? because of memory constraints
        
        '''
        
        with torch.inference_mode():
            logits= model(idx_cond)
            # print(logits.shape)
        logits = logits[-1, :] / temperature
        '''
        # print(logits.shape)
        # print(logits)
        
        # temperature what will do is that if temperature is high like 1.0 or more it will make the distribution more uniform and if temperature is low like 0.5 it will make the distribution more peaky means high probablity tokens will have more probablity and low probablity tokens will have even lower probablity so less randomness in output so optimum is around 0.7 to 0.9
        
        '''
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            '''
            # here v is top k which is arranged in descending order from big to small
            # print("v is :")
            # print(v)
            tensor([7.1562, 7.0312, 6.6875, 6.5625, 6.5625, 6.4375, 6.3750, 5.3438, 5.2812,
            4.9688, 4.9375, 4.8750, 4.8125, 4.7188, 4.6875, 4.6562, 4.5625, 4.3438,
            4.3438, 4.3438, 4.2500, 4.2188, 4.1875, 4.1875, 4.1562, 4.0938, 4.0625,
            4.0000, 3.9688, 3.9688, 3.9219, 3.7344, 3.7344, 3.7188, 3.7188, 3.6875,
            3.5000, 3.4531, 3.4219, 3.4062, 3.3750, 3.3125, 3.2969, 3.2812, 3.2500,
            3.2500, 3.2031, 3.1875, 3.1562, 3.1562], device='cuda:0',
            dtype=torch.bfloat16)
            '''
            # print(v[[-1]]) #tensor([3.1562], device='cuda:0', dtype=torch.bfloat16)
            
            logits[logits < v[[-1]]] = -float('Inf')
            '''
            # now take the last value of top k (3.1562) which is the smallest value among top k, set it as a cutoff  and set all the logits( total 201088) and those which are less than that to the lowest top k value i.e (eg 3.1526) will be converted to -inf so that when we apply softmax those will become zero probablity
            logits < v[[-1]]  # creates a boolean mask where True indicates logits less than the cutoff
            # tensor([ True, False, True, False, True])
            now here ## logits[logits < v[[-1]]] = -float('Inf')  For every element in logits 
            that satisfies means which is less than the last value --> which became true-->  logits < v[[-1]], replaced it with -∞.   
            '''
            '''
            # print(logits)
            tensor([4., -inf, -inf,  ..., -inf, -inf, -inf], device='cuda:0', dtype=torch.bfloat16)
            # print(logits.shape) #torch.Size([201088])
            '''
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        '''
        Suppose your vocabulary has 5 tokens and you get:
        probs = tensor([0.50, 0.20, 0.15, 0.10, 0.05])
        Now run:
        idx_next = torch.multinomial(probs, num_samples=1)
        print(idx_next)
        50% of the time → idx_next = tensor([0])
        20% of the time → idx_next = tensor([1])
        15% of the time → idx_next = tensor([2])
        etc.
        So it randomly picks one token index, but biased by probability.
        
        idx_next = torch.argmax(probs)
        the model would always choose the single highest-probability token — deterministic output (no randomness).
        That’s called greedy decoding.

        But with torch.multinomial, generation becomes stochastic (randomized) — allowing creativity, diversity, and more natural language.
        '''
        # print(idx_next)
        idx = torch.cat((idx, idx_next), dim=0)
        print(idx)

    
    # Decode and return
    result = token_ids_to_text(idx,tokenizer)
    
    '''
    idx is the promt converted to token like "context = "paris located in"" converted to tensor([ 796,  276, 7567,  306,   11], device='cuda:0')  and then with each iteration one by one till max token length tokens are added like 
        tensor([ 796,  276, 7567,  306,   11,  326], device='cuda:0')
        tensor([ 796,  276, 7567,  306,   11,  326,  316], device='cuda:0')
        tensor([ 796,  276, 7567,  306,   11,  326,  316,  395], device='cuda:0')
        tensor([ 796,  276, 7567,  306,   11,  326,  316,  395,  484], device='cuda:0')
        till 100 tokens are generated and added to the idx tensor
        .
        .
        .
        tensor([ 796,  276, 7567,  306,   11,  326,  316,  395,  484,   13,  364,  481,
        261,  623,  316, 1058, 1770,  328,   11, 1225,   11, 2059,   11,  328,
        261,  261, 1058,  326, 7557,   13, 1202,  501,   13, 1202,  326, 1869,
        11,  261,  364, 1335,   11,  483,  364,   13,  328, 1232,  290, 3627,
        11, 1225, 2059,   13,  261, 1023,   11,  673,  889,   13, 3627,  316,
        5981, 1354,  261, 3389,   11, 1770,  413, 2059,   13, 2059,  326,  290,
        5108,  328, 1869,   13, 1202,   11, 1679,  316,   11, 1354,   13, 3627,
        13,  364,  290, 1458,  364,  326, 2059,   13, 1679,  326,  480, 5981,
        13,  623,  501,   11,  316, 2059,  261,  673], device='cuda:0')
    
    '''
    
    return result



'''
'''

# context = "paris located in"
# # this prompt has 4 tokens 796, 276, 7567, 306 and starts from 0,1,2,3 and will generate 15 more tokens by the model beacause max_tokens=15 by default

# generate_text(model, context)


# a = torch.randn(4, 4,8)
# b= torch.rand(4,201088)

# b1 = b[-1,:]# it means last row of b
# b1.shape
# idx_cond = a[-2:]
# idx_cond.shape
# min(50, b.size(-1))

# z, _ = torch.topk(b, min(50, b.size(-1)))
