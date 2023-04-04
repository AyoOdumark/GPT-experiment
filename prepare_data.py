# 1. Implement Byte-pair encoding 

from tokenizers import pre_tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import Lowercase
from typing import List


class PrepareData:
    def __init__(self, path: str, train_size: float):
        self.path: str = path
        self.train_size: float = train_size
        self.corpus: List[str] = self._read_file()
        self.train_data, self.test_data = self._train_test_split()
        
    def get_train_data(self) -> List[str]:
        return self.train_data
    
    def get_test_data(self) -> List[str]:
        return self.test_data
        
    def _read_file(self):
        corpus = []
        with open(self.path, "r") as f:
            file = f.read()
        list_of_lines = file.splitlines()
        for sentence in list_of_lines:
            if sentence != "":
                corpus.append(sentence)
                
        return corpus
                
    def _train_test_split(self):
        train_data = []
        test_data = []
        corpus_length = len(self.corpus)
        train_idx = int(corpus_length * self.train_size)
        train_data = train_data + self.corpus[0:train_idx]
        test_data = test_data + self.corpus[train_idx + 1:]
        
        return train_data, test_data
        

def train_tokenizer(vocab_size: int, special_tokens: List, data: List[str]):
    normalizer = Lowercase()
    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(), Punctuation()])
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, special_tokens=special_tokens)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.train_from_iterator(iterator=data, trainer=trainer, length=len(data))
    print("Saving tokenizer as tokenizer.json...")
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved successfully!")
    
def load_tokenizer(path: str):
    return Tokenizer.from_file(path)

