# -*- coding: utf-8 -*-

import os
import argparse
import random

import numpy as np
import torch
from datasets import load_dataset

from angle_emb import AnglE, AngleDataTokenizer
from angle_emb.utils import logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True,
                    help='Specify model name or path to set transformer backbone, required')
parser.add_argument('--load_mlm_model', type=int, default=0, choices=[0, 1],
                    help='Specify load_mlm_model, choices [0, 1], defaut 0')
parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                    help='Specify tokenizer name or path. Default None, will use model_name_or_path')
parser.add_argument('--pretrained_model_path', type=str, default=None,
                    help='Specify pretrained model path to load pretrained model, default None')
parser.add_argument('--pretrained_lora_path', type=str, default=None,
                    help='Specify pretrained lora path to load lora, default None')
parser.add_argument('--train_name_or_path', type=str, required=True,
                    help='Specify huggingface datasets name or local file path for train set, required')
parser.add_argument('--train_subset_name', type=str, default=None,
                    help='Specify huggingface datasets subset name for train set, default None')
parser.add_argument('--train_split_name', type=str, default='train',
                    help='Specify huggingface datasets split name for train set, default `train`')
parser.add_argument('--valid_name_or_path', type=str, default=None,
                    help='Specify huggingface datasets name or local file path for valid set, default None.')
parser.add_argument('--valid_subset_name', type=str, default=None,
                    help='Specify huggingface datasets subset name for valid set, default None')
parser.add_argument('--valid_split_name', type=str, default='train',
                    help='Specify huggingface datasets split name for valid set, default `train`')
parser.add_argument('--valid_name_or_path_for_callback', type=str, default=None,
                    help='Specify huggingface datasets name or local file path for callback valid set. '
                         'The dataset format should be `DatasetFormats.A`. Default None.')
parser.add_argument('--valid_subset_name_for_callback', type=str, default=None,
                    help='Specify huggingface datasets subset name for valid set for callback use, default None')
parser.add_argument('--valid_split_name_for_callback', type=str, default='train',
                    help='Specify huggingface datasets split name for valid set for callback use, default `train`')
parser.add_argument('--prompt_template', type=str, default=None,
                    help='Specify prompt_template like "xxx: {text}", default None.'
                         'This prompt will be applied for all text columns.'
                         'If you want to specify different prompts for different text columns,'
                         'please handle it in the preprocessing step.')
parser.add_argument('--fix_data', type=int, default=1, choices=[0, 1],
                    help='Whether fix data (only works when prompt_template is not None), choices [0, 1], defaut 1')
parser.add_argument('--filter_duplicate', type=int, default=1, choices=[0, 1],
                    help='Specify filter_duplicate, choices [0, 1], defaut 1')
parser.add_argument('--save_dir', type=str, default=None,
                    help='Specify save dir, default None')
parser.add_argument('--seed', type=int, default=-1,
                    help='Specify random seed, default -1')
parser.add_argument('--dataset_seed', type=int, default=None,
                    help='Specify dataset random seed, default None')
parser.add_argument('--workers', type=int, default=2,
                    help='Specify dataset workers, default 2')
parser.add_argument('--cosine_w', type=float, default=0.0,
                    help='Specify weight for cosine loss, default 0.0')
parser.add_argument('--ibn_w', type=float, default=30.0,
                    help='Specify weight for ibn loss, default 30.0')
parser.add_argument('--angle_w', type=float, default=1.0,
                    help='Specify weight for angle loss, default 1.0')
parser.add_argument('--angle_tau', type=float, default=20.0,
                    help='Specify angle_tau, default 20.0')
parser.add_argument('--cosine_tau', type=float, default=20.0,
                    help='Specify cosine_tau, defaut 20.0')
parser.add_argument('--ibn_tau', type=float, default=20.0,
                    help='Specify ibn_tau, defaut 20.0')
parser.add_argument('--apply_lora', type=int, default=0, choices=[0, 1],
                    help='Specify lora flag, choices [0, 1], default 0')
parser.add_argument('--load_kbit', type=int, default=None, choices=[4, 8, 16],
                    help='Specify kbit training, choices [4, 8, 16], default None')
parser.add_argument('--lora_r', type=int, default=32,
                    help='Specify lora_r, defaut 32')
parser.add_argument('--lora_alpha', type=int, default=32,
                    help='Specify lora_alpha, defaut 32')
parser.add_argument('--lora_dropout', type=float, default=0.1,
                    help='Specify lora_dropout, defaut 0.1')
parser.add_argument('--lora_target_modules', type=str, default=None,
                    help='Specify lora_target_modules. comma serves as the splitter, such as `W,b`. Defaut None')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='Specify learning_rate, defaut 1e-5')
parser.add_argument('--warmup_steps', type=int, default=100,
                    help='Specify warmup_steps, defaut 100')
parser.add_argument('--logging_steps', type=int, default=100,
                    help='Specify logging_steps, defaut 100')
parser.add_argument('--pooling_strategy', type=str, default='cls',
                    help='Specify pooling_strategy from [`cls`, `last`, `avg`, `cls_avg`, `max`], default `cls`')
parser.add_argument('--epochs', type=int, default=10, help='Specify epochs, default 10')
parser.add_argument('--max_steps', type=int, default=-1,
                    help='Specify max steps, default -1 (Automatically calculated from epochs)')
parser.add_argument('--save_steps', type=int, default=100, help='Specify save_steps, default 1000')
parser.add_argument('--save_total_limit', type=int, default=1, help='Specify save_total_limit, default 1')
parser.add_argument('--save_strategy', type=str, default='steps', choices=['steps', 'epoch', 'no'],
                    help='Specify save_strategy, default steps')
parser.add_argument('--eval_steps', type=int, default=1000, help='Specify eval_steps, default 1000')
parser.add_argument('--evaluation_strategy', type=str, default='steps', choices=['steps', 'epoch', 'no'],
                    help='Specify evaluation_strategy, default steps')
parser.add_argument('--batch_size', type=int, default=32, help='Specify batch size, default 32')
parser.add_argument('--maxlen', type=int, default=512, help='Specify max length, default 512')
parser.add_argument('--streaming', action='store_true', default=False,
                    help='Flag to enable streaming mode, default False')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Specify gradient_accumulation_steps, default 1')
parser.add_argument('--torch_dtype', type=str, default=None, choices=['auto', 'float32', 'float16', 'bfloat16'],
                    help='Specify torch_dtype from [`auto`, `float32`, `float16`, `bfloat16`], default None')
parser.add_argument('--fp16', type=bool, default=None, choices=[0, 1],
                    help='Specify fp16, choices [0, 1], default None')
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_private_repo', type=int, default=1, choices=[0, 1],
                    help='Specify hub_private_repo, default 1')
parser.add_argument('--hub_model_id', type=str, default=None,
                    help='Specify hub_model_id, default None, format like organization/model_id')
# configure tokenizer
parser.add_argument('--tokenizer_padding', type=str, default="longest", choices=['longest', 'max_length'],
                    help='Specify tokenizer padding from [`longest`, `max_length`], default `longest`')
parser.add_argument('--tokenizer_padding_side', type=str, default=None, choices=['left', 'right'],
                    help='specify tokenizer padding side from [`left`, `right`], default None')
# configure LLM
parser.add_argument('--is_llm', type=int, default=0, choices=[0, 1],
                    help='Specify is_llm, choices [0, 1], defaut 0')
parser.add_argument('--apply_billm', type=int, default=0, choices=[0, 1],
                    help='Specify apply_billm, choices [0, 1], defaut 0')
parser.add_argument('--billm_model_class', type=str, default=None,
                    help='Specify billm model class name, default None')
# configure ESE
parser.add_argument('--apply_ese', type=int, default=0, choices=[0, 1],
                    help='Specify apply_ese to support Espresso Sentence Embedding training, default 0')
parser.add_argument('--ese_kl_temperature', type=float, default=1.0,
                    help='Specify KL temperature for ese, default 1.0')
parser.add_argument('--ese_compression_size', type=int, default=128,
                    help='Specify compression size for ese, default 128')
# configure teacher alignment
parser.add_argument('--teacher_name_or_path', type=str, default=None,
                    help='Specify model_name_or_path for teacher alignment, default None')
parser.add_argument('--teacher_pooling_strategy', type=str, default='cls',
                    help='Specify pooling strategy for teacher from [`cls`, `last`, `avg`, `cls_avg`, `max`], default `cls`')  # NOQA
# configure coword_random_mask_rate
parser.add_argument('--coword_random_mask_rate', type=float, default=0,
                    help='Specify coword_random_mask_rate, default 0')
# configure wandb
parser.add_argument('--wandb_project', type=str, default=None, help='Specify WANDB_PROJECT, default None')
parser.add_argument('--wandb_log_model', type=str, default=None, help='Specify WANDB_LOG_MODEL, default None')
args = parser.parse_args()
logger.info(f'Args: {args}')

if args.seed is not None and args.seed > 0:
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if args.wandb_project is not None:
    import wandb

    logger.info('Set up wandb...')
    os.environ['WANDB_PROJECT'] = args.wandb_project
    os.environ['WANDB_LOG_MODEL'] = args.wandb_log_model

    wandb.login()

if args.torch_dtype == 'float32':
    args.torch_dtype = torch.float32
elif args.torch_dtype == 'float16':
    args.torch_dtype = torch.float16
elif args.torch_dtype == 'bfloat16':
    args.torch_dtype = torch.bfloat16

apply_bfloat16 = None
if args.torch_dtype == torch.bfloat16:
    apply_bfloat16 = True

lora_config = {
    'r': args.lora_r,
    'lora_alpha': args.lora_alpha,
    'lora_dropout': args.lora_dropout,
}
if args.lora_target_modules is not None:
    lora_config['target_modules'] = [v.strip() for v in args.lora_target_modules.split(',') if v.strip()]

if args.coword_random_mask_rate > 0 and not args.load_mlm_model:
    logger.info('Detect coword_random_mask_rate > 0, automattically set load_mlm_model to 1')
    args.load_mlm_model = 1


def main():
    model = AnglE(args.model_name_or_path,
                  tokenizer_name_or_path=args.tokenizer_name_or_path,
                  max_length=args.maxlen,
                  pretrained_model_path=args.pretrained_model_path,
                  pretrained_lora_path=args.pretrained_lora_path,
                  pooling_strategy=args.pooling_strategy,
                  train_mode=True,
                  apply_lora=args.apply_lora,
                  lora_config_kwargs=lora_config,
                  load_kbit=args.load_kbit,
                  torch_dtype=args.torch_dtype,
                  apply_bfloat16=apply_bfloat16,
                  tokenizer_padding_side=args.tokenizer_padding_side,
                  is_llm=args.is_llm,
                  apply_billm=args.apply_billm,
                  billm_model_class=args.billm_model_class,
                  load_mlm_model=args.load_mlm_model)

    if os.path.exists(args.train_name_or_path):
        ds = load_dataset('json',
                          data_files=[args.train_name_or_path],
                          num_proc=args.workers,
                          streaming=args.streaming)
    else:
        ds = load_dataset(args.train_name_or_path,
                          args.train_subset_name,
                          num_proc=args.workers,
                          streaming=args.streaming)

    logger.info('Dataset overview:')
    print(ds)
    logger.info('Processing train...')
    if args.streaming:
        train_ds = ds[args.train_split_name].shuffle(args.dataset_seed).map(
            AngleDataTokenizer(model.tokenizer, model.max_length,
                               prompt_template=args.prompt_template,
                               fix_data=args.fix_data),
            num_proc=args.workers)
    else:
        train_ds = ds[args.train_split_name].shuffle(args.dataset_seed).map(
            AngleDataTokenizer(model.tokenizer, model.max_length,
                               prompt_template=args.prompt_template,
                               fix_data=args.fix_data),
            num_proc=args.workers)

    valid_ds = None
    if valid_ds is None and args.valid_name_or_path is not None:
        logger.info('Validation detected, processing validation...')
        if os.path.exists(args.valid_name_or_path):
            valid_ds = load_dataset('json', data_files=[args.valid_name_or_path], num_proc=args.workers)
        else:
            if args.valid_subset_name is not None:
                valid_ds = load_dataset(args.valid_name_or_path, args.valid_subset_name, num_proc=args.workers)
            else:
                valid_ds = load_dataset(args.valid_name_or_path, num_proc=args.workers)
        valid_ds = valid_ds[args.valid_split_name or 'train'].map(
            AngleDataTokenizer(model.tokenizer, model.max_length,
                               prompt_template=args.prompt_template,
                               fix_data=args.fix_data),
            num_proc=args.workers)

    valid_ds_for_callback = None
    if valid_ds_for_callback is None and args.valid_name_or_path_for_callback is not None:
        logger.info('Validation for callback detected, processing validation...')
        if os.path.exists(args.valid_name_or_path_for_callback):
            valid_ds_for_callback = load_dataset(
                'json', data_files=[args.valid_name_or_path_for_callback], num_proc=args.workers)
        else:
            if args.valid_subset_name_for_callback is not None:
                valid_ds_for_callback = load_dataset(
                    args.valid_name_or_path_for_callback,
                    args.valid_subset_name_for_callback,
                    num_proc=args.workers)
            else:
                valid_ds_for_callback = load_dataset(
                    args.valid_name_or_path_for_callback, num_proc=args.workers)
        valid_ds_for_callback = valid_ds_for_callback[args.valid_split_name_for_callback or 'train'].map(
            AngleDataTokenizer(model.tokenizer, model.max_length,
                               prompt_template=args.prompt_template,
                               fix_data=args.fix_data),
            num_proc=args.workers)

    argument_kwargs = {}
    if args.push_to_hub:
        assert args.hub_model_id is not None, 'Please specify hub_mode_id via --hub_model_id xxx'
        argument_kwargs['push_to_hub'] = True
        argument_kwargs['hub_private_repo'] = bool(args.hub_private_repo)
        argument_kwargs['hub_model_id'] = args.hub_model_id
    if args.wandb_project is not None:
        argument_kwargs['report_to'] = 'wandb'
    if args.max_steps > 0:
        argument_kwargs['max_steps'] = args.max_steps

    trainer_kwargs = None
    if args.teacher_name_or_path is not None:
        trainer_kwargs = {
            'teacher_name_or_path': args.teacher_name_or_path,
            'teacher_pooling_strategy': args.teacher_pooling_strategy,
        }
    if args.apply_ese:
        trainer_kwargs = trainer_kwargs or {}
        trainer_kwargs = dict(trainer_kwargs, **{
            'ese_kl_temperature': args.ese_kl_temperature,
            'ese_compression_size': args.ese_compression_size,
        })

    model.fit(
        train_ds=train_ds,
        valid_ds=valid_ds,
        valid_ds_for_callback=valid_ds_for_callback,
        output_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_kwargs={
            'cosine_w': args.cosine_w,
            'ibn_w': args.ibn_w,
            'angle_w': args.angle_w,
            'cosine_tau': args.cosine_tau,
            'ibn_tau': args.ibn_tau,
            'angle_tau': args.angle_tau,
        },
        fp16=args.fp16,
        filter_duplicate=args.filter_duplicate,
        argument_kwargs=argument_kwargs,
        apply_ese=args.apply_ese,
        trainer_kwargs=trainer_kwargs,
        coword_random_mask_rate=args.coword_random_mask_rate,
        padding=args.tokenizer_padding,
    )


if __name__ == '__main__':
    main()
