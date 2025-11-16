# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

import re
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
# model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

sequence_Example = "A E T C Z A O A A E"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')

'''
>>> encoded_input
{'input_ids': tensor([[ 0,  5,  9, 11, 23, 24,  5, 24,  5,  5,  9,  2]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
>>> 

0 is start and 2 is end token
'''