import torch
import torch.nn as nn
from attention import MultiHeadAttention

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, model_dim, width_factor=4, dropout_probability=0.1):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.linear1 = nn.Linear(model_dim, width_factor * model_dim)
        self.linear2 = nn.Linear(width_factor * model_dim, model_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_probability)
        
    def forward(self, input_tensors):
        output = self.linear1(input_tensors)
        output = self.gelu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        
        return output
    
class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, dropout_probability=0.1):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embeddings = nn.Embedding(sequence_length, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_probability)
        
    def forward(self, input_idx):
        _, seq_len = input_idx.shape
        word_embeds = self.word_embeddings(input_idx)
        word_embeds = self.dropout(word_embeds)
        
        pos_embeds = self.positional_embeddings(torch.arange(seq_len).to(device))
        pos_embeds = self.dropout(pos_embeds)
        
        return word_embeds + pos_embeds
         
class TransformerBlock(nn.Module):
    def __init__(self, sequence_length, embedding_dim, num_of_heads, dropout_probability):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_of_heads, sequence_length, dropout_probability)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.positionwise_feedforward_net = PositionWiseFeedForwardNet(embedding_dim)
        self.dropout = nn.Dropout(p=dropout_probability)
    
    def forward(self, embeds):
        # Multihead attention
        output = self.multi_head_attention(embeds)
        # Residual Connection
        output = output + embeds
        # Layer normalization
        output = self.layer_norm(output)
        # Pointwise Feedforward network
        residual = output
        residual = self.dropout(residual)
        output = self.positionwise_feedforward_net(output)
        # Residual connection
        output = output + residual
        # Layer normalization
        output = self.layer_norm(output)
        
        return output
        

