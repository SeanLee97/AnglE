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
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_zhnli'], help='Specify mode from [`train`], default `train`')
parser.add_argument('--task', type=str, default='STS-B', help='Specify task from [`NLI-STS`, `STS-B`]')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='Specify pretrained_model_path, default None')
parser.add_argument('--pretrained_lora_path', type=str, default=None, help='Specify pretrained_lora_path, default None')
parser.add_argument('--save_dir', type=str, default=None, help='Specify save dir, default None')
parser.add_argument('--seed', type=int, default=42, help='Specify random seed, default 42')
parser.add_argument('--load_kbit', type=int, default=None, choices=[4, 8, 16], help='Specify load_kbit, default None')
parser.add_argument('--workers', type=int, default=25, help='Specify dataset workers, default None')
parser.add_argument('--w1', type=float, default=1.0, help='Specify w1 (cosine), default 1.0')
parser.add_argument('--w2', type=float, default=1.0, help='Specify w2 (ibn), default 1.0')
parser.add_argument('--w3', type=float, default=1.0, help='Specify w3 (angle), default 1.0')
parser.add_argument('--angle_tau', type=float, default=1.0, help='Specify angle_tau, default 1.0')
parser.add_argument('--cosine_tau', type=float, default=20.0, help='Specify cosine_tau, defaut 20.0')
parser.add_argument('--ibn_tau', type=float, default=20.0, help='Specify ibn_tau, defaut 20.0')
parser.add_argument('--lora_r', type=int, default=32, help='Specify lora_r, defaut 32')
parser.add_argument('--lora_alpha', type=int, default=32, help='Specify lora_alpha, defaut 32')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='Specify lora_dropout, defaut 0.1')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Specify learning_rate, defaut 1e-5')
parser.add_argument('--warmup_steps', type=int, default=100, help='Specify warmup_steps, defaut 100')
parser.add_argument('--pooling_strategy', type=str, default='cls',
                    help='Specify pooling_strategy from [`avg`, `cls`, `cls_avg`, `first_last_avg`]')
parser.add_argument('--epochs', type=int, default=20, help='Specify epochs, default 20')
parser.add_argument('--save_steps', type=int, default=100, help='Specify save_steps, default 100')
parser.add_argument('--eval_steps', type=int, default=None, help='Specify eval_steps, default None')
parser.add_argument('--batch_size', type=int, default=32, help='Specify batch size, default 32')
parser.add_argument('--maxlen', type=int, default=512, help='Specify max length, default 512')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Specify gradient_accumulation_steps, default 1')
parser.add_argument('--do_eval', type=int, default=1, choices=[0, 1], help='Specify do_eval, default 1')
parser.add_argument('--fp16', type=bool, default=None, choices=[0, 1], help='Specify fp16, default None')
parser.add_argument('--debug_sample_size', type=int, default=None, help='Specify debug_sample_size, default None')
parser.add_argument('--compute_similar_matrix', type=int, default=1, choices=[0, 1], help='Specify compute_similar_matrix, default 1')
parser.add_argument('--model_name', type=str, default='NousResearch/Llama-2-7b-hf',
                    help='Specify model_name, default NousResearch/Llama-2-7b-hf')
args = parser.parse_args()
print('Args:', args)

if args.seed is not None:
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

TASK_MAPPING = {
    'NLI-STS': 'nli-sts',
    'STS-B': ('mteb/stsbenchmark-sts', ),
    'MRPC': ('SetFit/mrpc', ),
    'QQP': ('SetFit/qqp', ),
    'QNLI': ('SetFit/qnli', ),
    'RTE': ('SetFit/rte', ),
    'ZHNLI': 'shibing624/nli-zh-all',
}
assert args.task in TASK_MAPPING
task_name = TASK_MAPPING[args.task]

PROMPT = 'Summarize sentence "{text}" in one word:"' if 'llama' in args.model_name.lower() else None


def load_data(split_name):
    if args.task in ['MRPC', 'QQP', 'QNLI', 'RTE']:
        col1, col2 = 'text1', 'text2'
    else:
        col1, col2 = 'sentence1', 'sentence2'
    if args.task in ['QNLI', 'RTE']:
        label_mapping = {
            '1': 0, # 'not entailment'
            '0': 1, # entailment
        }
    else:
        label_mapping = {}
    data = [
        {'text1': obj[col1], 'text2': obj[col2],
         'label': float(obj['score']) if args.task == 'STS-B' else int(label_mapping.get(str(obj['label']), obj['label']))}
        for obj in load_dataset(*task_name)[split_name]
    ]
    return data


def load_nli_data(exclude_neutral=True):
    def load_all_nli():
        label_mapping = {
            'entailment': 1,  # '0' (entailment)
            'neutral': 1,
            'contradiction': 0   # '2' (contradiction)
        }
        data = []
        with gzip.open('./data/AllNLI.tsv.gz', 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'train' and row['label'] != 'neutral':
                    if exclude_neutral and row['label'] == 'neutral':
                        continue
                    sent1 = row['sentence1'].strip()
                    sent2 = row['sentence2'].strip()
                    data.append({'text1': sent1, 'text2': sent2, 'label': label_mapping[row['label']]})
        return data

    def load_sts():
        all_sts = []
        for i in range(12, 17):
            all_sts += [
                {'text1': obj['sentence1'],
                 'text2': obj['sentence2'],
                 'label': float(obj['score'])}
                for obj in load_dataset(f"mteb/sts{i}-sts")['test']
            ]
        all_sts += [
            {'text1': obj['sentence1'],
             'text2': obj['sentence2'],
             'label': float(obj['score'])}
            for obj in load_dataset(f"mteb/stsbenchmark-sts")['test']
        ]
        all_sts += [
            {'text1': obj['sentence1'],
             'text2': obj['sentence2'],
             'label': float(obj['score'])}
            for obj in load_dataset(f"mteb/sickr-sts")['test']
        ]
        return all_sts

    return load_all_nli(), load_sts()


def load_zhnli():
    ds = {}
    all_data = []
    for name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']:
        data = load_dataset('shibing624/nli_zh', name)
        data = data.rename_column("sentence1", "text1")
        data = data.rename_column("sentence2", "text2")
        
        new_data = []
        for obj in data['test']:
            # if name == 'STS-B':
            #    obj['label'] /= 5.
            new_data.append(obj)
        all_data += new_data
        ds[name] = Dataset.from_list(new_data)
    return DatasetDict(ds), all_data


train_data, valid_data, test_data = None, None, None
to_dataset = True
if args.task == 'NLI-STS':
    train_data, test_data = load_nli_data()
    valid_data = test_data
    print('train size:', len(train_data))
    print('test size:', len(test_data))
elif args.task == 'ZHNLI':
    train_data = [obj for obj in load_dataset('shibing624/nli-zh-all')['train']]
    zh_nli_ds, valid_data = load_zhnli()
else:
    train_data, valid_data, test_data = [
        load_data(split) for split in ['train', 'validation', 'test']
    ]
    if args.task in ['QQP', 'QNLI', 'RTE']:
        print('>>> QQP or QNLI eval at validation data')
        test_data = valid_data
    print('train size:', len(train_data))
    print('test size:', len(test_data))

# to Dataset
if to_dataset:
    dataset = {}
    if train_data is not None:
        if args.debug_sample_size is not None:
            print(f'>>> debug: sample_size={args.debug_sample_size}')
            train_data = train_data[:args.debug_sample_size]
        train_ds = Dataset.from_list(train_data)
        dataset['train'] = train_ds
    if valid_data is not None:
        valid_ds = Dataset.from_list(valid_data)
        dataset['validation'] = valid_ds
    if test_data is not None:
        test_ds = Dataset.from_list(test_data)
        dataset['test'] = test_ds
    dataset = DatasetDict(dataset)

if args.mode == 'train':
    print('train mode...')
    # build model
    if 'llama' in args.model_name.lower() or 'qwen' in args.model_name.lower():
        print('loading llama...')
        model = AnglE(args.model_name,
                      max_length=args.maxlen,
                      apply_lora=True,
                      pooling_strategy=args.pooling_strategy,
                      lora_config_kwargs={
                          'r': args.lora_r,
                          'lora_alpha': args.lora_alpha,
                          'lora_dropout': args.lora_dropout,
                          'target_modules': ['q_proj', 'v_proj']},
                      train_mode=True,
                      is_llm=True,
                      pretrained_lora_path=args.pretrained_lora_path,
                      pretrained_model_path=args.pretrained_model_path,
                      load_kbit=args.load_kbit)
    else:
        PROMPT = None
        model = AnglE(args.model_name,
                      max_length=args.maxlen,
                      apply_lora=False,
                      pooling_strategy=args.pooling_strategy,
                      train_mode=True)
    
    print('>>> PROMPT:', PROMPT)
    train_ds = dataset['train'].shuffle().map(AngleDataTokenizer(model.tokenizer, model.max_length, prompt_template=PROMPT), num_proc=args.workers)
    if args.do_eval:
        valid_ds = dataset['validation'].map(AngleDataTokenizer(model.tokenizer, model.max_length, prompt_template=PROMPT), num_proc=args.workers)
    else:
        valid_ds = None
    
    model.fit(
        train_ds=train_ds,
        valid_ds=valid_ds,
        output_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if args.do_eval == 1 and args.eval_steps is not None else None,
        warmup_steps=args.warmup_steps,
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
elif args.mode == 'test':
    print('test...')
    model = AnglE.from_pretrained(
        args.model_name,
        pretrained_model_path=args.save_dir,
        pretrained_lora_path=args.pretrained_lora_path,
        pooling_strategy=args.pooling_strategy,
        load_kbit=args.load_kbit,
        max_length=args.maxlen,
    ).cuda()
    test_ds = dataset['test'].map(AngleDataTokenizer(model.tokenizer, model.max_length, prompt_template=PROMPT), num_proc=args.workers)
    corrcoef, accuracy = model.evaluate(test_ds, batch_size=args.batch_size, device=model.device)
    print(f'corrcoef: {corrcoef}, accuracy: {accuracy}')
elif args.mode == 'test_zhnli':
    model = AnglE.from_pretrained(
        args.model_name,
        pretrained_model_path=args.save_dir,
        pretrained_lora_path=args.pretrained_lora_path,
        pooling_strategy=args.pooling_strategy,
        load_kbit=args.load_kbit,
        max_length=args.maxlen,
    ).cuda()
    all_corrcoef = []
    for dataset in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']:
        print(f'eval {dataset}...')
        test_ds = zh_nli_ds[dataset].map(AngleDataTokenizer(model.tokenizer, model.max_length, prompt_template=PROMPT), num_proc=args.workers)
        corrcoef, accuracy = model.evaluate(test_ds, batch_size=args.batch_size, device=model.device)
        print(f'{dataset}: corrcoef: {corrcoef}, accuracy: {accuracy}')
        all_corrcoef.append(corrcoef)
    print('avg corrcoef:', sum(all_corrcoef)/len(all_corrcoef))
else:
    raise ValueError(f'not support {args.mode}')
