import torch 
import torch.nn as nn
from transformer import TransformerBlock
from transformer import Embeddings

class GPT_1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_of_layers, seq_length, num_of_heads, dropout_probability):
        super(GPT_1, self).__init__()
        self.num_of_layers = num_of_layers
        self.embeddings = Embeddings(vocab_size, embedding_dim, seq_length)
        self.transformer = nn.ModuleList([TransformerBlock(seq_length, embedding_dim, num_of_heads, dropout_probability) 
                                                 for _ in range(num_of_layers)])
        # self.transformer_block = TransformerBlock(seq_length, embedding_dim, num_of_heads, dropout_probability)
        
        
    def forward(self, input):
        embeds = self.embeddings(input)
        output = embeds
        for block in self.transformer:
            output = block(output)
            
        return output
    
