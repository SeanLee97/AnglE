# -*- coding: utf-8 -*-

import os
import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from angle_emb import AnglE, AngleDataTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Specify pretrained_model_path, default None')
parser.add_argument('--pretrained_lora_path', type=str, default=None, help='Specify pretrained_lora_path, default None')
parser.add_argument('--bellm_class_name', type=str, default=None, help='Specify bellm class name, default None')
parser.add_argument('--train_name_or_path', type=str, required=True, help='Specify train_name_or_path, required')
parser.add_argument('--train_subset_name', type=str, default=None, help='Specify train_subset_name, required')
parser.add_argument('--prompt', type=str, default=None, help='Specify prompt')
parser.add_argument('--save_dir', type=str, default=None, help='Specify save dir, default None')
parser.add_argument('--seed', type=int, default=42, help='Specify random seed, default 42')
parser.add_argument('--dataset_seed', type=int, default=None, help='Specify random dataset_seed, default None')
parser.add_argument('--workers', type=int, default=25, help='Specify dataset workers, default None')
parser.add_argument('--w1', type=float, default=1.0, help='Specify w1 (cosine), default 1.0')
parser.add_argument('--w2', type=float, default=35.0, help='Specify w2 (ibn), default 1.0')
parser.add_argument('--w3', type=float, default=1.0, help='Specify w3 (angle), default 1.0')
parser.add_argument('--angle_tau', type=float, default=1.0, help='Specify angle_tau, default 1.0')
parser.add_argument('--cosine_tau', type=float, default=20.0, help='Specify cosine_tau, defaut 20.0')
parser.add_argument('--ibn_tau', type=float, default=20.0, help='Specify ibn_tau, defaut 20.0')
parser.add_argument('--is_llm', type=int, default=0, choices=[0, 1], help='Specify is_llm, defaut 1')
parser.add_argument('--apply_lora', type=int, default=0, choices=[0, 1], help='Specify apply_lora, defaut 0')
parser.add_argument('--load_kbit', type=int, default=None, choices=[4, 8, 16], help='Specify load_kbit, default None')
parser.add_argument('--lora_r', type=int, default=32, help='Specify lora_r, defaut 32')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora_alpha, defaut 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora_dropout, defaut 0.1')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify learning_rate, defaut 1e-5')
parser.add_argument('--start_bilayer_index', type=int, default=None, help='Specify start_bilayer_index, defaut None')
parser.add_argument('--warmup_steps', type=int, default=100, help='Specify warmup_steps, defaut 100')
parser.add_argument('--logging_steps', type=int, default=100, help='Specify logging_steps, defaut 100')
parser.add_argument('--pooling_strategy', type=str, default='cls',
                    help='Specify pooling_strategy from [`avg`, `cls`, `cls_avg`, `first_last_avg`]')
parser.add_argument('--epochs', type=int, default=20, help='Specify epochs, default 20')
parser.add_argument('--save_steps', type=int, default=100, help='Specify save_steps, default 1000')
parser.add_argument('--batch_size', type=int, default=32, help='Specify batch size, default 32')
parser.add_argument('--maxlen', type=int, default=512, help='Specify max length, default 512')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Specify gradient_accumulation_steps, default 1')
parser.add_argument('--torch_dtype', type=str, default=None, help='Specify torch_dtype, default 1')
parser.add_argument('--fp16', type=bool, default=None, choices=[0, 1], help='Specify fp16, default None')
parser.add_argument('--compute_similar_matrix', type=int, default=1, choices=[0, 1], help='Specify compute_similar_matrix, default 1')
parser.add_argument('--push_to_hub', type=int, default=1, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_model_id', type=str, default=None, help='Specify push_to_hub_model_id, default None, format like organization/model_id')
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
              is_llm=args.is_llm,
              apply_lora=args.apply_lora,
              lora_config_kwargs={
                  'r': args.lora_r,
                  'lora_alpha': args.lora_alpha,
                  'lora_dropout': args.lora_dropout,
                  'target_modules': ['fc2', 'Wqkv', 'fc1'] if 'BePhi2Model' == args.bellm_class_name else None,
              },
              load_kbit=args.load_kbit,
              bellm_class_name=args.bellm_class_name,
              kbit_kwargs={'use_gradient_checkpointing': False} if 'BePhi2Model' == args.bellm_class_name else None,
              torch_dtype=args.torch_dtype)


if args.start_bilayer_index is not None:
    model.backbone.set_start_bilayer_index(args.start_bilayer_index)

if os.path.exists(args.train_name_or_path):
    ds = load_dataset('json', data_files=[args.train_name_or_path])
else:
    ds = load_dataset(args.train_name_or_path, args.train_subset_name)

print(ds)
train_ds = ds['train'].shuffle(args.dataset_seed).map(AngleDataTokenizer(model.tokenizer, model.max_length, prompt_template=args.prompt), num_proc=args.workers)


argument_kwargs = {}
if args.push_to_hub:
    assert args.hub_model_id is not None
    argument_kwargs['push_to_hub'] = True,
    argument_kwargs['hub_model_id'] = args.hub_model_id


model.fit(
    train_ds=train_ds,
    output_dir=args.save_dir,
    batch_size=args.batch_size,
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    save_steps=args.save_steps,
    warmup_steps=args.warmup_steps,
    logging_steps=args.logging_steps,
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
    argument_kwargs=argument_kwargs
)
