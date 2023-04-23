from tokenizers import pre_tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import Lowercase
from time import sleep
from typing import List

class BytePairTokenizer:
    def __init__(self, vocab_size: int, special_tokens="[UNK]") -> None:
        self.normalizer = Lowercase()
        self.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(), Punctuation()])
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True, special_tokens=[special_tokens])
        self.tokenizer.normalizer = self.normalizer
        self.tokenizer.pre_tokenizer = self.pre_tokenizer
        
    def train(self, file_path: str, save_to: str) -> None:
        print("Training tokenizer...")
        self.tokenizer.train(files=[file_path], trainer=self.trainer)
        print("Training complete.")
        print(f"Saving tokenizer as {save_to}...")
        self.tokenizer.save(save_to)
        sleep(2)
        print("Tokenizer saved successfully!")
        
class BBPETokenizer:
    def __init__(self, vocab_size: int, special_tokens=["<|endoftext|>"]):
        self.tokenizer = Tokenizer(BPE())
        self.trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        
    def train(self, file_path: str, save_to: str) -> None:
        print("Training tokenizer...")
        self.tokenizer.train([file_path], trainer=self.trainer)
        print("Training complete.")
        print(f"Saving tokenizer as {save_to}...")
        self.tokenizer.save(save_to)
        sleep(2)
        print("Tokenizer saved successfully!")
        
def load_tokenizer(path: str):
    return Tokenizer.from_file(path)

