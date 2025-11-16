

import torch
import pandas as pd


data = pd.read_csv(r'/home/dipayan/Documents/nano-gpt-oss/datasets/test_data.csv')


list = data['0'].to_list()
list = list[:10]


list[0]
b = 'MCGIFGYCNYLVERSRGEIIDTLVDGLQRLEYRGY'

from transformers import AutoTokenizer, AutoModelForMaskedLM

import re
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
# model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

sequence_Example = "A E T C Z A O A A E"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')



# sequence_Example = list[0]
sequence_Example = b
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='pt')



