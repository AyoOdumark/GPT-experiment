## GPT-experiments

This repo contains experiments on the GPT paradigm. The attention models were implemented from scratch for educational purposes so we expect a performance drop compared to pytorch's implementation. Still under active developments, but you can play around with `playground.py` file to train your own GPT model. 

Colab Playground to run this repo

https://colab.research.google.com/drive/1ihFsG_pO1hKyKwsdaayyEtPD-vwoRzyE?usp=sharing

## Installations

Dependencies:
- [pytorch](https://pytorch.com)
- [numpy](https://numpy.org/install)
- `pip install tokenizers` for huggingface tokenizers (our tokenizer interface was built on it)
- `pip install wandb` for logging of metrics and visualization

## Quick Start

To get started, we advise you take a look at the `config.py` file to see the list of arguments that the `playground.py` file accepts. The `playground.py` is the training file.

## Tokenizer 

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

We do not advise you train the model on CPU. While this is not impossible, we strongly advise you use a GPU. To train GPU, use the following arguments so the model can train on GPU. This will result in a smaller model than the OpenAI GPT models but it still gets the job done

```
$ python playground.py path_to_corpus --vocab_size=size-of-vocabulary --embedding_dim=384 --num_of_layers=4 --num_of_heads=4 --batch_size=16 --tokenizer=path-to-tokenizer
```

This command will train a mini GPT model with 4 transformer layers with 4 heads in each layer. For the learning rate scheduler, we implemented a cosine annealing scheduler with warmup of 2000 iterations in accordance with the original paper on GPT-1. We save the model checkpoint as `model_checkpoint.tar` every 500 iterations by default (this can be changed). The default dropout rate is 0.1 but you can change to `--dropout=0.0` for a smaller model. We print the validation loss every 500 iterations

## Evaluation and Generation

This is a work in progress. The GPT model definition in `gpt.py` has a generate function that based on top-k sampling. I am yet to write the `generate.py` file. The generate function is usable for generating text, and you can get it to work.

## Baselines

Baseline reporting will be here!

## Todos

- Create baselines
- Finish up the generate.py
- Allow config files in yaml for easier training and evaluations
- Implement l2 regularization on non-bias weights as stated in GPT-1 paper
- Try other learning rate scheduler
- Generalize attention model to include BERT implementation
- Work on finetuning models for other tasks
- Train model on a cluster
- Investigation other initialization. Currently, I use the default pytorch init, and not the init scheme given in the paper.


## Acknowledgments

I want to thank about team at [rectlabs](https://rectlabs.com).

## Collaborations

Feel free to reach out at [mail](ayomide@rectlabs.com). Looking for collaborators in Africa to work with. 