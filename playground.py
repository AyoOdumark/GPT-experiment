# 1. Load tokenizer
# 2. Play around with the concept of dataloader and batch tokenization
# 3. Run a sample train loop while configuring every part of the GPT experiment

# TODO
# 1. Create a learning rate scheduler. Use AdamW optimizer instead of Adam
# 2. Write evaluation and generation function to see output of model
# 3. Optimize the DataLoader by using pytorch dataloader or add multiprocessing to the current one
# 4. Make the hyperparameters into a dataclass so as to make them flexible

import wandb
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import DataLoader, Dataset
from preprocessing import preprocessing
from preprocessing.tokenizer import load_tokenizer
from gpt import GPT_1
from config import Config

device = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # Use Gradient scaler and mixed precision

class CorpusDataset(Dataset):
    def __init__(self, data, context_size):
        self.data = data
        self.n_data = len(data)
        self.context_size = context_size
        
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        X = torch.LongTensor([self.data[idx: idx + self.context_size]])
        y = torch.LongTensor([self.data[idx + 1: idx + 1 + self.context_size]])
        
        return X, y

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
            
def evaluate(model, X_val, y_val, criterion):
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            y_pred = model(X_val)
            val_loss = criterion(y_pred.view(-1, y_pred.size(-1)), y_val.view(-1))
            return val_loss
    
def main(opt):
    gpt = GPT_1(opt.vocab_size, opt.embedding_dim, opt.num_of_layers, opt.context_size, opt.num_of_heads, opt.dropout_proba)
    corpus = preprocessing.read_file(opt.corpus_path)
    
    train, test = preprocessing.split_dataset(corpus, train_size=0.8)
    bpe_tokenizer = load_tokenizer(opt.tokenizer)
    
    train_token_ids = preprocessing.tokenize_and_encode(train, bpe_tokenizer)
    test_token_ids = preprocessing.tokenizer_and_encode(test, bpe_tokenizer)
    
    train_dataset = CorpusDataset(train_token_ids, opt.context_size)
    test_dataset = CorpusDataset(test_token_ids, opt.context_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, pin_memory=True)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(gpt.parameters(), lr=opt.learning_rate)

    # Adding GradScaler and Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Refactor this
    
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            y_pred = gpt(x) 
            loss = criterion(y_pred.view(-1, y_pred.size(-1)), y.view(-1)) 
            loss = loss / opt.num_accumulation_steps
    
        # Accumulates scaled gradients
        scaler.scale(loss).backward()
        
        if (i + 1) % opt.num_accumulation_steps == 0:
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Evaluation
        X_val, y_val = next(iter(test_dataloader))
        X_val = X_val.to(device, non_blocking=True)
        y_val = y_val.to(device, non_blocking=True)
        val_loss = evaluate(gpt, X_val, y_val, criterion=criterion)
                
        print(f"Train loss: {loss.item()}      Val loss: {val_loss.item()}")
            
        wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    opt = Config().parse()
    wandb.login()
    wandb.init(project=opt.wandb_name)
    main(opt)

