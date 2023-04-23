## GPT-experiments

This repo contains experiments on the GPT paradigm. The attention models were implemented from scratch for educational purposes so we expect a performance drop compared to pytorch's implementation. Still under active developments, but you can play around with `playground.py` file to train your own GPT model. 

Colab Playground to run this repo

https://colab.research.google.com/drive/1ihFsG_pO1hKyKwsdaayyEtPD-vwoRzyE?usp=sharing

## install

Dependencies:
- [pytorch](https://pytorch.com)
- [numpy](https://numpy.org/install)
- `pip install tokenizers` for huggingface tokenizers (our tokenizer interface was built on it)
- `pip install wandb` for logging of metrics and visualization

## quick start

To get started, we advise you take a look at the `config.py` file to see the list of arguments that the `playground.py` file accepts. The `playground.py` is the training file.

## Training Tokenizer 

We advise you train tokenizer on your own corpus using  `train_tokenizer.py` in the `preprocessing` folder. The `train_tokenizer.py` receives three arguments: `corpus_path` argument which is the path to your corpus (we expect a .txt file); `vocab_size` for the size of unique tokens in vocabulary; `tokenizer` which can be `bpe` for vanilla Byte-Pair Encoding and `bbpe` for Byte-level Byte Pair Encoding which was used in the GPT-2 and GPT-3 papers; the last argument `--out` is an optional argument which is to specify where to save the tokenizer.

To train Byte-Pair Encoding tokenizer, do the following:
1. Change directory to preprocessing.

```
$ cd preprocessing
```

2. Run the following

```
$ python train_tokenizer.py corpus_path bpe 
```

After training the tokenizer will save in the preprocessing folder which is the current working directory as `tokenizer.json`. 

## Training