# -*- coding: utf-8 -*-

import os
import json
import gzip
import csv
import argparse
import random

import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from angle_emb import AnglE, AngleDataTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Specify pretrained_model_path, default None')
parser.add_argument('--pretrained_lora_path', type=str, default=None, help='Specify pretrained_lora_path, default None')
parser.add_argument('--train_path', type=str, required=True, help='Specify train path, required')
parser.add_argument('--save_dir', type=str, default=None, help='Specify save dir, default None')
parser.add_argument('--seed', type=int, default=42, help='Specify random seed, default 42')
parser.add_argument('--workers', type=int, default=25, help='Specify dataset workers, default None')
parser.add_argument('--w1', type=float, default=1.0, help='Specify w1 (cosine), default 1.0')
parser.add_argument('--w2', type=float, default=35.0, help='Specify w2 (ibn), default 1.0')
parser.add_argument('--w3', type=float, default=1.0, help='Specify w3 (angle), default 1.0')
parser.add_argument('--angle_tau', type=float, default=1.0, help='Specify angle_tau, default 1.0')
parser.add_argument('--cosine_tau', type=float, default=20.0, help='Specify cosine_tau, defaut 20.0')
parser.add_argument('--ibn_tau', type=float, default=20.0, help='Specify ibn_tau, defaut 20.0')
parser.add_argument('--apply_lora', type=int, default=0, choices=[0, 1], help='Specify apply_lora, defaut 0')
parser.add_argument('--lora_r', type=int, default=32, help='Specify lora_r, defaut 32')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora_alpha, defaut 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora_dropout, defaut 0.1')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify learning_rate, defaut 1e-5')
parser.add_argument('--warmup_steps', type=int, default=100, help='Specify warmup_steps, defaut 100')
parser.add_argument('--pooling_strategy', type=str, default='cls',
                    help='Specify pooling_strategy from [`avg`, `cls`, `cls_avg`, `first_last_avg`]')
parser.add_argument('--epochs', type=int, default=20, help='Specify epochs, default 20')
parser.add_argument('--save_steps', type=int, default=100, help='Specify save_steps, default 1000')
parser.add_argument('--batch_size', type=int, default=32, help='Specify batch size, default 32')
parser.add_argument('--maxlen', type=int, default=512, help='Specify max length, default 512')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Specify gradient_accumulation_steps, default 1')
parser.add_argument('--fp16', type=bool, default=None, choices=[0, 1], help='Specify fp16, default None')
parser.add_argument('--compute_similar_matrix', type=int, default=1, choices=[0, 1], help='Specify compute_similar_matrix, default 1')
parser.add_argument('--model_name', type=str, default='roberta-large',
                    help='Specify model_name, default roberta-large')
args = parser.parse_args()
print('Args:', args)

if args.seed is not None and args.seed > 0:
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

        
model = AnglE(args.model_name,
              max_length=args.maxlen,
              pretrained_model_path=args.pretrained_model_path,
              pretrained_lora_path=args.pretrained_lora_path,
              pooling_strategy=args.pooling_strategy,
              train_mode=True,
              apply_lora=args.apply_lora,
              lora_config_kwargs={
                  'r': args.lora_r,
                  'lora_alpha': args.lora_alpha,
                  'lora_dropout': args.lora_dropout,
              })

ds = load_dataset('json', data_files=[args.train_path])

print(ds)
train_ds = ds['train'].shuffle().map(AngleDataTokenizer(model.tokenizer, model.max_length), num_proc=args.workers)

model.fit(
        train_ds=train_ds,
        output_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=100,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_kwargs={
            'w1': args.w1,
            'w2': args.w2,
            'w3': args.w3,
            'cosine_tau': args.cosine_tau,
            'ibn_tau': args.ibn_tau,
            'angle_tau': args.angle_tau,
        },
        fp16=args.fp16,
    )
