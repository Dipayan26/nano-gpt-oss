import torch,gc
from torch.utils.data import Dataset,DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset
# from tqdm.notebook import tqdm
from tqdm import tqdm
batch_size=5
context_len=10

#____________________________________
#____________________________________
import pandas as pd
data = pd.read_csv(r'/home/dipayan/Documents/nano-gpt-oss/datasets/test_data.csv')
list = data['0'].to_list()
list = list[:10]
#____________________________________

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

import re
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
# model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")

sequence_Example = "A E T C Z A O A A E"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example)
#____________________________________

dataset = load_dataset("roneneldan/TinyStories")

# dataset2 = dataset['train'][0]['text'][:10]
# dataset2 = dataset['train'][0]


#____________________________________________
train_text = "".join(ex for ex in list[:5])
val_text = "".join(ex for ex in list[5:])
#____________________________________________

train_text = " ".join([ex["text"] for ex in dataset['train']])
val_text = " ".join([ex["text"] for ex in dataset['validation']])

# train_text = " ".join([ex["text"] for ex in dataset['train']])
# val_text = " ".join([ex["text"] for ex in dataset['validation']])

dataset['train']['text'][0]
train_text[0:1000]
# train_text[0:10]

tokenizer = get_tokenizer()
print("tokenizing...")

# train_tokens = tokenizer.encode(train_text[0:10])
train_tokens = tokenizer.encode(train_text[:100])
val_tokens = tokenizer.encode(val_text[:100])
print("tokenized")


# len(train_tokens)
# len(val_tokens)

# print(range(0, len(train_tokens) - 8192, 8192))
#max_length= context_len = 10 not 8192
class TextDataset(Dataset):
    def __init__(self, tokens, max_length=8192, stride=8192):
        self.input_ids = []
        self.target_ids = []
        for i in tqdm(range(0, len(tokens) - max_length, stride)):
            
            # [0:0+4000]
            input_chunk = tokens[i:(i + max_length)]
            # print('Input chunk tokens:', len(input_chunk))  # Debugging statement
            # [0+1:0+4000+1]
            target_chunk = tokens[(i + 1):(i + max_length + 1)]
            # print('Target chunk tokens:', len(target_chunk))  # Debugging statement
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
val_dataset = TextDataset(val_tokens, max_length=context_len, stride=context_len)

# train_dataset[0]
# train_dataset[1]
# train_dataset[2]
# val_dataset[0]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


gc.get_stats()
del dataset, train_text, val_text
gc.collect()



