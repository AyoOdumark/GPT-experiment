# 1. Load tokenizer
# 2. Play around with the concept of dataloader and batch tokenization
# 3. Run a sample train loop while configuring every part of the GPT experiment

# TODO
# 1. Create a learning rate scheduler
# 2. Write evaluation and generation function to see output of model
# 3. Optimize the DataLoader by using pytorch dataloader or add multiprocessing to the current one
# 4. Make the hyperparameters into a dataclass so as to make them flexible

import wandb
import torch 
import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
from prepare_data import PrepareData
from prepare_data import load_tokenizer
from gpt import GPT_1

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Weight and biases
wandb.login()

wandb.init(project="GPT1-exp")

# Hyperparameters
VOCAB_SIZE = 10000 
EMBEDDING_DIM = 768
NUM_OF_LAYERS = 12 # change to 6 if the GPU runs out of memory
SEQ_LENGTH = 512
NUM_OF_HEADS = 12
DROP_PROBABILITY = 0.1
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
CONTEXT_SIZE = 512
NUM_ACCUMULATION_STEPS = 4 

gpt1 = GPT_1(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, num_of_layers=NUM_OF_LAYERS, 
            seq_length=SEQ_LENGTH, num_of_heads=NUM_OF_HEADS, dropout_probability=DROP_PROBABILITY).to(device)

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

# for token_id in range(1, len(train_token_ids)):

class NaiveDataLoader:
    def __init__(self, data, context_size, batch_size):
        self.data = data
        self.n_data = len(data)
        self.context_size = context_size
        self.batch_size = batch_size
        self.n_batches = int(self.n_data / self.batch_size)
        
    def __len__(self):
        return self.batch_size
    
    def get_batch(self):
        for _ in range(self.n_batches):
            idx = torch.randint(len(self.data) - self.context_size, (self.batch_size,))
            X = torch.LongTensor([self.data[i: i+self.context_size] for i in idx])
            y = torch.LongTensor([self.data[i+1: i+1+self.context_size] for i in idx])
            
            yield X, y
    
train_dataloader = NaiveDataLoader(train_token_ids, CONTEXT_SIZE, BATCH_SIZE)
train_iter = iter(train_dataloader.get_batch())

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(gpt1.parameters(), lr=LEARNING_RATE)

# Refactor this
def train():
    
    for i, (x, y) in enumerate(train_iter):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        y_pred = gpt1(x)
        
        loss = criterion(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        
        loss = loss / NUM_ACCUMULATION_STEPS
        loss.backward()
        
        if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_iter)):
            # Update weights
            optimizer.step()
        
        print(f"Train loss: {loss.item()}")
        
        wandb.log({"loss": loss.item()})
        

if __name__ == "__main__":
    train()
    

