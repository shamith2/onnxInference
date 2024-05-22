# Script to train tokenizer on tinyStories dataset
# Adapted from https://huggingface.co/docs/tokenizers/en/quicktour

import os

import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tinyStories')

if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)

# load dataset
dataset = load_dataset('roneneldan/TinyStories')

# write data to csv
data_file_csv = os.path.join(dir, 'stories.csv')
data_file_txt = os.path.join(dir, 'stories.txt')

concatenate_datasets([dataset['train'], dataset['validation']]).to_csv(data_file_csv)

# convert csv file to txt file
df = pd.read_csv(data_file_csv)

with open(data_file_txt, 'w') as file:
    for row in df['text']:
        file.write(str(row) + '\n')

# initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token='<|unknown|>'))
trainer = BpeTrainer(special_tokens=['<|unknown|>', '<|im_start|>', '<|im_end|>'], vocab_size=2048, min_frequency=1)
tokenizer.pre_tokenizer = Whitespace()

# train and save tokenizer
tokenizer.train([data_file_txt], trainer)
tokenizer.save(os.path.join(dir, 'tinyTokenizer.json'))
