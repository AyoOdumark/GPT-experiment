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
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input):
        embeds = self.embeddings(input)
        output = embeds
        for block in self.transformer:
            output = block(output)
        
        return nn.functional.log_softmax(self.linear(output), dim=-1)
    
    @torch.no_grad()
    def generate(self, input_context, max_token, topk):
        # This implements the top-k sampling scheme
        predicted_tokens = []
        for _ in range(max_token):
            input_tensor = torch.LongTensor([input_context])
            # print(input_tensor.shape)

            # Get the output 
            y_pred = self.forward(input_tensor)

            # Crop the last prediction
            y_pred = y_pred[:, -1, :]
    
            v, _ = torch.topk(y_pred, k=topk)
            y_pred[y_pred < v[:, [-1]]] = -float("Inf")
    
            probs = nn.functional.softmax(y_pred)
            idx = torch.multinomial(probs, num_samples=1)
            
            predicted_tokens.append(idx.item())
            
            input_context.append(idx.item())
            input_context = input_context[1:]
            
        return predicted_tokens
    
    
