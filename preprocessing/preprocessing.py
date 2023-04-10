# 1. Implement Byte-pair encoding 
from typing import List

def read_file(path: str) -> List[str]:
    # accepts path to the corpus and returns a list of strings. Each element of the list is a line in the text file
    corpus = []
    with open(path, "r") as f:
        file = f.read()
    list_of_lines = file.splitlines()
    for sentence in list_of_lines:
        if sentence != "":
            corpus.append(sentence)
            
    return corpus

def read_multiple_files(list_of_paths: List[str]) -> List[str]:
    pass

def split_dataset(dataset: List[str], train_size: float):
    # If encode is true, this function will return the token ids in the train and test data. 
    train = []
    test = []
    dataset_length = len(dataset)
    train_idx = int(dataset_length * train_size)
    train_data = train_data + dataset[0:train_idx]
    test_data = test_data + dataset[train_idx + 1:]
    
    return train, test
    
def tokenize_and_encode(dataset: List[str], tokenizer):
    token_ids = []
    
    for sentence in dataset:
        output = tokenizer.encode(sentence)
        token_ids += output.ids
        
    return token_ids

