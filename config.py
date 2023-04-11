# This should contain the CLI.
# Basic settings such as dimension states, number of layers, and so on
# The GPT config such be a dataclass

import argparse

class ConfigBase:
    def __init__(self):
        self.name = argparse.Namespace()
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    def initialization(self):
        self.parser.add_argument("corpus_path", type=str, 
                            help="Corpus file. Should be a txt file.")
        self.parser.add_argument("--vocab_size", type=int, default=10000, 
                            help="Vocabulary size. Vocabulary is created with byte-level byte pair encoding")
        self.parser.add_argument("--learning_rate", type=float, default=1e-5, help="starting learning rate")
        self.parser.add_argument("--Embedding_dim", type=int, default=768, 
                            help="Embedding dimension. Make sure this argument is a multiple of attention heads")
        self.parser.add_argument("--num_of_layers", type=int, default=12, help="Number of transformer blocks")
        self.parser.add_argument("--num_of_heads", type=int, default=12, 
                            help="Number of attention heads. Make sure this argument is a divisor of Embedding dimension")
        self.parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
        self.parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        self.parser.add_argument("--context_size", type=int, default=512, help="Size of context")
        self.parser.add_argument("--num_accumulation_steps", type=int, default=4, help="Gradient Accumulation steps")
        self.parser.add_argument("--dropout_proba", type=float, default=0.1, 
                            help="Dropout probability. Value must be between 0 and 1.")
        self.parser.add_argument("--wandb_name", type=str, default="GPT-exp", help="This is for logging purposes")
        self.parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer")
        
    def _parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


class Config(ConfigBase):
    def __init__(self):
        super().__init__()
        self.initialization()
        
    def parse(self):
        opt = self._parse()
        
        return opt

