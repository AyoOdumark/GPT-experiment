import torch 
import torch.nn as nn
from transformer import TransformerBlock
from transformer import Embeddings

class miniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length, num_of_heads, dropout_probability):
        super(miniGPT, self).__init__()
        self.embeddings = Embeddings(vocab_size, embedding_dim, seq_length)
        self.transformer_block = TransformerBlock(seq_length, embedding_dim, num_of_heads, dropout_probability)
        
        
    def forward(self, input):
        embeds = self.embeddings(input)
        output = self.transformer_block(embeds)
        output = self.transformer_block(output)
        
        return output
    
