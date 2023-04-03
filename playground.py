# 1. Load tokenizer
# 2. Play around with the concept of dataloader and batch tokenization
# 3. Run a sample train loop while configuring every part of the GPT experiment

import torch 
from torch.utils.data import DataLoader, Dataset
from prepare_data import PrepareData
from prepare_data import load_tokenizer
from gpt import GPT_1

gpt1 = GPT_1(vocab_size=10000, embedding_dim=768, num_of_layers=12, seq_length=512, num_of_heads=12, dropout_probability=0.1)

data_path = "Brown.txt"
prepare = PrepareData(path=data_path, train_size=0.8)


tokenizer_path = "tokenizer.json"
tokenizer = load_tokenizer("tokenizer.json")

# Creating a data loader
train = prepare.get_train_data()
test = prepare.get_test_data()

train_token_ids = []
for sentence in train:
    output = tokenizer.encode(sentence)
    train_token_ids += output.ids
    

# output = gpt1(torch.LongTensor([train_token_ids[:512], train_token_ids[1:513]]))

batch = 1
# for token_id in range(1, len(train_token_ids)):

