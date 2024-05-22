# Script to train models on tinyStories daatset
# dataset: https://huggingface.co/datasets/roneneldan/TinyStories
# Adapted from https://github.com/broskicodes/slms/blob/master/small_lms_train.py

import os
import sys
import gc
import logging

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler

from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from babyLlama import babyLlama, ModelArgs
from babyDecoder import babyDecoder

model_name = 'babyDecoder'
dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)

if not os.path.exists(dir):
    os.makedirs(dir, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dir, model_name + '.log'), filemode='w', level=logging.INFO)

# setup device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# if device == torch.device('cuda:0'):
#     memtrace_filename = model_name + '_memtrace'
#     torch.cuda.memory._record_memory_history()

#     torch.set_default_dtype(torch.bfloat16)

logger.info("using device: {}, dtype: {}".format(device, torch.get_default_dtype()))

# load dataset and tokenizer
dataset = load_dataset('roneneldan/TinyStories')

tokenizer_file = os.path.join('tinyStories', 'tinyTokenizer.json')
tokenizer = Tokenizer.from_file(tokenizer_file)

# out of memory error with this tokenizer
# could be because of large vocab size leading to large embedding layer
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M',
#                                           add_prefix_space=True)

# hyperparameters
hyperparameters = {
    "n_epochs": 5,
    "vocab_size": tokenizer.get_vocab_size(),
    "learning_rate": 5e-4,
    "beta1": 0.90,
    "beta2": 0.95,
    "weight_decay": 0.10
}

model_args: ModelArgs = ModelArgs(
    vocab_size=hyperparameters['vocab_size'],
    device=device,
    dim=1024,
    n_layers=4,
    n_heads=8,
    hidden_dim=768,
    max_batch_size=8,
    max_seq_len=1024
)

# parameters and hyperparameters
n_epochs = hyperparameters['n_epochs']
vocab_size = hyperparameters['vocab_size']
max_seq_len = model_args.max_seq_len
batch_size = model_args.max_batch_size
learning_rate = hyperparameters['learning_rate']
betas = (hyperparameters['beta1'], hyperparameters['beta2'])
weight_decay = hyperparameters['weight_decay']
dim = model_args.dim
n_heads = model_args.n_heads
n_layers = model_args.n_layers

logger.info('vocab size: {}'.format(vocab_size))

# tokenize dataset
# add pad token
# tokenizer.pad_token = tokenizer.eos_token
# eos_token_id = tokenizer.encode(tokenizer.eos_token)[0]

# def train_tok_fn(train_data):
#     return tokenizer.encode(train_data['text'], padding='max_length', truncation='longest_first',
#                             max_length=max_seq_len, is_split_into_words=True, return_tensors='pt')

# def val_tok_fn(val_data):
#     return tokenizer.encode(val_data['text'], padding='max_length', truncation='longest_first',
#                             max_length=max_seq_len, is_split_into_words=True, return_tensors='pt')

# train_tok_dataset = dataset['train'].map(lambda x: {'tokens': [data for data in train_tok_fn(x)]},
#                                          batched=True, remove_columns=['text']).with_format('torch')

# val_tok_dataset = dataset['validation'].map(lambda x: {'tokens': [data for data in val_tok_fn(x)]},
#                                             batched=True, remove_columns=['text']).with_format('torch')

# tokenize dataset
eos_token_id = 2
tokenizer.enable_padding(pad_id=eos_token_id, pad_token='<|im_end|>', length=max_seq_len)
tokenizer.enable_truncation(max_length=max_seq_len)

train_tok_dataset = dataset['train'].map(lambda x: {'tokens': [elem.ids for elem in tokenizer.encode_batch(x['text'])]},
                                         batched=True, remove_columns=['text']).with_format('torch').select(range(100000))

val_tok_dataset = dataset['validation'].map(lambda x: {'tokens': [elem.ids for elem in tokenizer.encode_batch(x['text'])]},
                                            batched=True, remove_columns=['text']).with_format('torch').select(range(5000))

# training and validation dataloader
train_dataloader = DataLoader(train_tok_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_tok_dataset, batch_size=batch_size, shuffle=True)

# model and trainer
# model = babyLlama(model_args).to(device)
model = babyDecoder(model_args).to(model_args.device)
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
num_params = sum(p.numel() for p in model.parameters()) // 1e6

num_training_steps = n_epochs * len(train_dataloader)
scheduler = lr_scheduler.ConstantLR(optimizer)

logger.info('model size: {}M parameters'.format(num_params))

# train model
loss_log = []
lr = []
progress_bar = tqdm(range(num_training_steps))

gc.collect()

if device == torch.device('cuda:0'):
    torch.cuda.empty_cache()

for epoch in range(n_epochs):
    # training mode
    model.train()

    train_losses = torch.zeros(len(train_dataloader), device=device)

    for j, t_batch in enumerate(train_dataloader):
        t_batch = t_batch['tokens'].to(device)
        t_targets = torch.concat((t_batch[:, 1:], eos_token_id * torch.ones([batch_size, 1]).to(device)), dim=-1).type_as(t_batch)

        # t_indices = torch.arange(0, max_seq_len, dtype=torch.int, device=device)

        t_logits = model(t_batch) # t_logits = model(t_batch, t_indices)

        bsz, seqlen, vsz = t_logits.shape
        t_logits = t_logits.view(bsz * seqlen, vsz)
        t_targets = t_targets.view(bsz * seqlen)

        t_loss = torch.nn.functional.cross_entropy(t_logits, t_targets)

        # gradients and backpropagation
        optimizer.zero_grad()
        t_loss.backward()

        optimizer.step()
        scheduler.step()
        progress_bar.update(1)

        if (progress_bar.n % 500 == 0):
            logger.info('sub-batch train loss: {}, lr: {}'.format(str(t_loss.item()), str(optimizer.param_groups[0]['lr'])))
        
        train_losses[j] = t_loss.item()
        loss_log.append(t_loss.log10().item())
        lr.append(optimizer.param_groups[0]['lr'])
    
    with torch.no_grad():
        # eval mode
        model.eval()

        val_losses = torch.zeros(len(val_dataloader), device=device)
        
        for k, v_batch in enumerate(val_dataloader):
            v_batch = v_batch['tokens'].to(device)
            v_targets = torch.concat((v_batch[:, 1:], eos_token_id * torch.ones([batch_size, 1]).to(device)), dim=-1).type_as(v_batch)

            # v_indices = torch.arange(0, max_seq_len, dtype=torch.int, device=device)

            v_logits = model(v_batch) # v_logits = model(v_batch, v_indices)

            bsz, seqlen, vsz = v_logits.shape
            v_logits = v_logits.view(bsz * seqlen, vsz)
            v_targets = v_targets.view(bsz * seqlen)

            v_loss = torch.nn.functional.cross_entropy(v_logits, v_targets)

            val_losses[k] = v_loss.item()
            predictions = torch.argmax(v_logits, dim=-1)

        avg_val_loss = val_losses.mean()
        
        logger.info('val loss: {}'.format(avg_val_loss))

    # save checkpoint
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_args": model_args,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "hyperparameters": hyperparameters,
        "val_loss": avg_val_loss,
        "train_loss": train_losses.mean(),
        "learning_rate": lr
    }

    checkpoint_file = os.path.join(dir, '{}-epoch-{}M-checkpoint-{}-{}.pt'
                                        .format(str(n_epochs), str(num_params), str(epoch),
                                                str(round(avg_val_loss.item(), 3))))
    
    logging.info('saving checkpoint in {}'.format(checkpoint_file))
    torch.save(checkpoint, checkpoint_file)

torch.save(torch.tensor(loss_log), os.path.join(dir, 'loss-log-' + str(num_params) + '.pt'))

# if device == torch.device('cuda:0'):
#     torch.cuda.memory._dump_snapshot(os.path.join(dir, memtrace_filename + '.pickle'))

plt.plot(torch.tensor(loss_log).view(-1, 25).mean(1))
plt.savefig(os.path.join(dir, 'loss-plot-' + str(num_params) + '.png'))
