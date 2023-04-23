from tokenizer import BytePairTokenizer, BBPETokenizer
from preprocessing import read_file
import argparse


# CLI parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("corpus_path", type=str, help="Path to corpus you want to train tokenizer on")
parser.add_argument("vocab_size", type=int, help="Vocabulary size.")
parser.add_argument("--tokenizer", type=str, default="bpe", help="Choose tokenization scheme. Write bbpe for byte-level BPE or bpe for vanilla BPE")
parser.add_argument("--out", type=str, default="tokenizer.json", help="output path to save the tokenizer after training")

opt = parser.parse_args()

# Create tokenizer
if opt.tokenizer == "bpe":
    tokenizer = BytePairTokenizer(opt.vocab_size)
elif opt.tokenizer == "bbpe":
    tokenizer = BBPETokenizer(opt.vocab_size)
else:
    print("Enter a valid tokenizer.")

# Train tokenizer and save as tokenizer.json
tokenizer.train(opt.corpus_path, opt.out)

