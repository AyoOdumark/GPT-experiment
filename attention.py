import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, model_dim, head_dim, seq_length, dropout_probability):
        super(SelfAttention, self).__init__()
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.W_k = nn.Linear(model_dim, head_dim, bias=False)
        self.W_q = nn.Linear(model_dim, head_dim, bias=False)
        self.W_v = nn.Linear(model_dim, head_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # self.register_buffer("tril", torch.tril(torch.ones(seq_length, seq_length)))
        
        self.attention_dropout = nn.Dropout(p=dropout_probability)
        
    def forward(self, input_embeddings):
        # batch_size, seq_length, embed_dim = input_embeddings
        queries = self.W_q(input_embeddings)
        keys = self.W_k(input_embeddings)
        values = self.W_v(input_embeddings)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) /self.head_dim**0.5
        
        # Masking 
        tril = torch.tril(torch.ones((scores.shape[-1], scores.shape[-1])))
        scores = scores.masked_fill(tril==0, float("-inf"))
        
        attention_weights = self.softmax(scores)
        
        attention_weights = self.attention_dropout(attention_weights)
        
        attention_vectors = torch.matmul(attention_weights, values)
        return attention_vectors
    

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_length, dropout_probability):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_of_heads == 0
        
        self.model_dim = model_dim
        self.head_dim = int(model_dim / num_of_heads)
        self.num_of_heads = num_of_heads
        
        self.attention_heads = nn.ModuleList(SelfAttention(self.model_dim, self.head_dim, seq_length, dropout_probability) 
                                             for _ in range(num_of_heads))
        self.W_o = nn.Linear(self.num_of_heads*self.head_dim, self.model_dim, bias=False)
        
        self.dropout = nn.Dropout(p=dropout_probability)
        
    def forward(self, input_embeddings):
        heads = [attn_head(input_embeddings) for attn_head in self.attention_heads]
        heads_concat = torch.cat(heads, dim=-1)
        attention_vectors = self.W_o(heads_concat)
        
        attention_vectors = self.dropout(attention_vectors)
        
        return attention_vectors
    

