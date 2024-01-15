# -*- coding: utf-8 -*-

import os
import re
import sys
import json
from functools import partial
from typing import Any, Dict, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass

import scipy
import scipy.stats
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from tqdm import tqdm
from boltons.iterutils import chunked_iter
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoModel, AutoTokenizer,
    PreTrainedModel, Trainer, TrainingArguments,
    TrainerCallback, BitsAndBytesConfig
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from peft import (
    get_peft_model, LoraConfig, TaskType, PeftModel,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training
)
from peft.tuners.lora import LoraLayer

from . import bellm
from .utils import logger


def set_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def find_all_linear_names(model: PreTrainedModel, linear_type: Optional[object] = None) -> List[str]:
    if linear_type is None:
        linear_type = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_type):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def categorical_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor, from_logits: bool = True):
    if from_logits:
        return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)
    return -(torch.log(y_pred, dim=1) * y_true).sum(dim=1)


def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0):
    # modified from: https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent.py#L79
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


def angle_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 1.0):
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()

    y_pred_re, y_pred_im = torch.chunk(y_pred, 2, dim=1)
    a = y_pred_re[::2]
    b = y_pred_im[::2]
    c = y_pred_re[1::2]
    d = y_pred_im[1::2]

    # (a+bi) / (c+di)
    # = ((a+bi) * (c-di)) / ((c+di) * (c-di))
    # = ((ac + bd) + i(bc - ad)) / (c^2 + d^2)
    # = (ac + bd) / (c^2 + d^2) + i(bc - ad)/(c^2 + d^2)
    z = torch.sum(c**2 + d**2, dim=1, keepdim=True)
    re = (a * c + b * d) / z
    im = (b * c - a * d) / z

    dz = torch.sum(a**2 + b**2, dim=1, keepdim=True)**0.5
    dw = torch.sum(c**2 + d**2, dim=1, keepdim=True)**0.5
    re /= (dz / dw)
    im /= (dz / dw)

    y_pred = torch.concat((re, im), dim=1)
    y_pred = torch.abs(torch.sum(y_pred, dim=1)) * tau  # absolute delta angle
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)


def in_batch_negative_loss(y_true: torch.Tensor,
                           y_pred: torch.Tensor,
                           tau: float = 20.0,
                           negative_weights: float = 0.0):
    """in-batch negative loss
    """
    device = y_true.device

    def make_target_matrix(y_true: torch.Tensor):
        idxs = torch.arange(0, y_pred.shape[0]).int().to(device)
        y_true = y_true.int()
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]

        idxs_1 *= y_true.T
        idxs_1 += (y_true.T == 0).int() * -2

        idxs_2 *= y_true
        idxs_2 += (y_true == 0).int() * -1

        y_true = (idxs_1 == idxs_2).float()
        return y_true

    neg_mask = make_target_matrix(y_true == 0)

    y_true = make_target_matrix(y_true)

    # compute similarity
    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T  # dot product
    similarities = similarities - torch.eye(y_pred.shape[0]).to(device) * 1e12
    similarities = similarities * tau

    if negative_weights > 0:
        similarities += neg_mask * negative_weights

    return categorical_crossentropy(y_true, similarities, from_logits=True).mean()


def contrastive_with_negative_loss(text: torch.Tensor, pos: torch.Tensor, neg: Optional[torch.Tensor] = None, tau: float = 20.0):
    target = torch.cat((pos, neg), dim=0)  if neg is not None else pos  # (2B, D)
    q_norm = torch.nn.functional.normalize(text, p=2, dim=1)  # (B, D)
    t_norm = torch.nn.functional.normalize(target, p=2, dim=1)  # (2B, D)
    scores = torch.mm(q_norm, t_norm.transpose(0, 1)) * tau  # (B, 2B)
    labels = torch.tensor(
        range(len(scores)), dtype=torch.long, device=scores.device
    )
    return nn.CrossEntropyLoss()(scores, labels)


def compute_corrcoef(x, y):
    return scipy.stats.spearmanr(x, y).correlation


def l2_normalize(vecs):
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def optimal_threshold(y_true, y_pred):
    loss = lambda t: -np.mean((y_true > 0.5) == (y_pred > np.tanh(t)))
    result = scipy.optimize.minimize(loss, 1, method='Powell')
    return np.tanh(result.x), -result.fun


class Prompts:
    A = 'Summarize sentence "{text}" in one word:"'
    B = 'You can only output one word. Summarize "{text}":"'
    C = 'Represent this sentence for searching relevant passages: {text}'

    @classmethod
    def list_prompts(cls):
        for key, val in Prompts.__dict__.items():
            if key.startswith('_') or key == 'list_prompts':
                continue
            print(f'Prompts.{key}', '=', f"'{val}'")


class DatasetFormats:
    '''
    format A: text1,text2,label
    input format: [
        text1[0],
        text2[0],
        text1[1],
        text2[1],
        ...
    ]
    label format: [
        label[0],
        label[0],
        label[1],
        label[1],
        ...
    ]
    '''
    A = 'text1,text2,label'

    '''
    format B: text,positive,negative
    input format: [
        text[0],
        positive[0],
        negative[0],
        text[1],
        positive[1],
        negative[1],
        ...
    ]
    '''
    B = 'text,positive,negative'

    @classmethod
    def list_formats(cls):
        for key, val in DatasetFormats.__dict__.items():
            if key.startswith('_') or key == 'list_formats':
                continue
            print(f'DatasetFormats.{key}', '=', f"'{val}'")


class AngleDataTokenizer:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: Optional[int] = None,
                 prompt_template: Optional[str] = None,
                 template_placeholders: Optional[List[str]] = None,
                 extra_columns: Optional[List[str]] = None,
                 dataset_format: Optional[str] = None,
                 end_with_eos: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.prompt_template_tok = None
        self.extra_columns = extra_columns
        self.dataset_format = dataset_format
        self.end_with_eos = end_with_eos
        if template_placeholders is None:
            template_placeholders = ['condition', 'text']
        if prompt_template is not None:
            re_placeholder = re.compile(r'\{(%s)\}' % '|'.join(template_placeholders))
            self.prompt_template_tok = self.tokenizer(re_placeholder.sub('', prompt_template))

    @staticmethod
    def fix_bad_data(token_ids, prompt_ids):
        bad_index = -1
        for idx in range(len(token_ids) - 1, -1, -1):
            try:
                bad_index = prompt_ids.index(token_ids[idx])
            except ValueError:
                break
        if bad_index == -1:
            return token_ids
        # print('bad index:', prompt_ids[bad_index])
        to_fix_ids = prompt_ids[bad_index:]
        return token_ids[:len(token_ids) - len(to_fix_ids)] + to_fix_ids

    def __call__(self, data: Dict) -> Dict:
        if self.dataset_format is None:
            if 'text1' in data and 'text2' in data and 'label' in data:
                logger.info(f'Detect DatasetFormats.A: {DatasetFormats.A}')
                self.dataset_format = DatasetFormats.A
            elif 'text' in data and 'positive' in data and 'negative' in data:
                self.dataset_format = DatasetFormats.B
                logger.info(f'Detect DatasetFormats.B: {DatasetFormats.B}')
            else:
                raise NotImplementedError('Currently only support two dataset formats'
                                          'DatasetFormats A: must include three columns: `text1`, `text2`, and `label`.'
                                          'DatasetFormats B: mut include three columns: `text`, `positive`, `negative`')
        text_columns = None
        if self.dataset_format == DatasetFormats.A:
            text_columns = ['text1', 'text2']
        elif self.dataset_format == DatasetFormats.B:
            text_columns = ['text', 'positive', 'negative']

        extra_length = 0
        extra_placeholder = {}
        if self.extra_columns is not None:
            for key, val in data.items():
                if key not in self.extra_columns:
                    continue
                extra_placeholder[key] = val
                extra_length += len(self.tokenizer(val, add_special_tokens=False)['input_ids'])
        if self.end_with_eos:
            extra_length += 1

        if self.prompt_template_tok is not None:
            for text_column in text_columns:
                tok = self.tokenizer(data[text_column],
                                     max_length=self.max_length - len(self.prompt_template_tok['input_ids']) - extra_length,
                                     truncation=True,
                                     add_special_tokens=False)
                data[text_column] = self.tokenizer.decode(tok['input_ids'])
                data[text_column] = self.prompt_template.format(text=data[text_column], **extra_placeholder)

        toks = []
        for text_column in text_columns:
            toks.append(self.tokenizer(data[text_column], max_length=self.max_length, truncation=True))

        if self.prompt_template_tok is not None:
            for tok in toks:
                if tok['input_ids'][-1] != self.prompt_template_tok['input_ids'][-1]:
                    print('bad data:', f"token ids: {tok['input_ids']}, prompt token ids: {self.prompt_template_tok['input_ids']}")
                    tok['input_ids'] = self.fix_bad_data(tok['input_ids'], self.prompt_template_tok['input_ids'])
                    try:
                        assert len(tok['input_ids']) == len(tok['attention_mask'])
                        assert tok['input_ids'][-1] == self.prompt_template_tok['input_ids'][-1]
                        print('fixed it ;)')
                        print('new data:', f"token ids: {tok['input_ids']}, prompt token ids: {self.prompt_template_tok['input_ids']}")
                    except AssertionError:
                        print('failed to fix it :()')

        combined_tok = {}
        seperate_ids = []
        for idx, tok in enumerate(toks):
            for key, val in tok.items():
                if idx == 0:
                    combined_tok[key] = val
                else:
                    combined_tok[key] += val
                if key == 'input_ids':
                    seperate_ids += [idx] * len(val)

        combined_tok['labels'] = [int(data['label']) if 'label' in data else -1]
        combined_tok['seperate_ids'] = seperate_ids
        combined_tok['extra'] = {
            'dataset_format': self.dataset_format,
            'end_with_eos': self.end_with_eos
        }
        return combined_tok


@dataclass
class AngleDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        has_token_type_ids = "token_type_ids" in features[0]
        end_with_eos = features[0]['extra']['end_with_eos']

        new_features = []
        for feature in features:
            seperate_ids = feature['seperate_ids']
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            assert len(seperate_ids) == len(input_ids) == len(attention_mask)

            has_token_type_ids = False
            if "token_type_ids" in feature:
                has_token_type_ids = True
                token_type_ids = feature['token_type_ids']
                assert len(token_type_ids) == len(input_ids)

            max_seperate_id = max(seperate_ids)
            prev_start_idx = 0
            for start_idx in range(1, max_seperate_id + 1):
                start_idx = seperate_ids.index(start_idx)

                new_feature = {}
                new_feature['input_ids'] = input_ids[prev_start_idx:start_idx]
                new_feature['attention_mask'] = attention_mask[prev_start_idx:start_idx]
                if has_token_type_ids:
                    new_feature['token_type_ids'] = token_type_ids[prev_start_idx:start_idx]
                new_feature['labels'] = feature['labels']
                new_features.append(new_feature)
                prev_start_idx = start_idx

            # last
            new_feature = {}
            new_feature['input_ids'] = input_ids[prev_start_idx:]
            new_feature['attention_mask'] = attention_mask[prev_start_idx:]
            if has_token_type_ids:
                new_feature['token_type_ids'] = token_type_ids[prev_start_idx:]
            new_feature['labels'] = feature['labels']
            new_features.append(new_feature)

        # remove features
        del features

        if end_with_eos:
            features = self.tokenizer.pad(
                {'input_ids': [feature['input_ids'] for feature in new_features]},
                padding=False,
                max_length=self.max_length - 1,
                return_tensors=return_tensors,
                truncation=True,
            )
            features['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in features['input_ids']]
            features = self.tokenizer.pad(features, padding=self.padding, return_tensors=return_tensors)
        else:
            features = self.tokenizer.pad(
                {'input_ids': [feature['input_ids'] for feature in new_features]},
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )
        features['attention_mask'] = self.tokenizer.pad(
            {'input_ids': [feature['attention_mask'] for feature in new_features]},
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )['input_ids']
        if has_token_type_ids:
            features['token_type_ids'] = self.tokenizer.pad(
                {'input_ids': [feature['token_type_ids'] for feature in new_features]},
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )['input_ids']
        features['labels'] = torch.Tensor([feature['labels'] for feature in new_features])

        return features


class Pooler:
    def __init__(self,
                 model: PreTrainedModel,
                 pooling_strategy: Optional[str] = None,
                 padding_strategy: Optional[str] = None,
                 is_llm: bool = False):
        self.model = model
        self.pooling_strategy = pooling_strategy
        self.padding_strategy = padding_strategy
        self.is_llm = is_llm

    def __call__(self, inputs) -> Any:
        if self.is_llm or self.pooling_strategy == 'last':
            batch_size = inputs['input_ids'].shape[0]
            if self.padding_strategy == 'left':
                sequence_lengths = -1
            else:
                sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1

        if self.is_llm:
            hidden_states = self.model(output_hidden_states=True, return_dict=True, **inputs).hidden_states[-1]
            outputs = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        else:
            outputs = self.model(**inputs).last_hidden_state
            if self.pooling_strategy == 'cls':
                outputs = outputs[:, 0]
            elif self.pooling_strategy == 'cls_avg':
                outputs = (outputs[:, 0] + torch.mean(outputs, dim=1)) / 2.0
            elif self.pooling_strategy == 'last':
                outputs = outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
            elif self.pooling_strategy == 'avg':
                outputs = torch.mean(outputs, dim=1)
            elif self.pooling_strategy == 'max':
                outputs, _ = torch.max(outputs, dim=1)
            else:
                raise NotImplementedError('please specify pooling_strategy from [`cls`, `last`, `avg`, `max`, `last_avg`]')
        return outputs


class AngleTrainer(Trainer):
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = DatasetFormats.A,
                 **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_fct = AngleLoss(dataset_format=dataset_format, **loss_kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        outputs = self.pooler(inputs)
        loss = self.loss_fct(labels, outputs)
        return (loss, outputs) if return_outputs else loss


class AngleLoss:
    def __init__(self,
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0,
                 cosine_tau: float = 20.0,
                 ibn_tau: float = 20.0,
                 angle_tau: float = 1.0,
                 dataset_format: Optional[str] = None,
                 **kwargs):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.cosine_tau = cosine_tau
        self.ibn_tau = ibn_tau
        self.angle_tau = angle_tau
        self.dataset_format = dataset_format

    def __call__(self,
                 labels: torch.Tensor,
                 outputs: torch.Tensor) -> torch.Tensor:
        if self.dataset_format == DatasetFormats.A:
            loss = 0.
            if self.w1 > 0:
                loss += self.w1 * cosine_loss(labels, outputs, self.cosine_tau)
            if self.w2 > 0:
                loss += self.w2 * in_batch_negative_loss(labels, outputs, self.ibn_tau)
            if self.w3 > 0:
                loss += self.w3 * angle_loss(labels, outputs, self.angle_tau)
        elif self.dataset_format == DatasetFormats.B:
            # text,positive,negative
            text = outputs[::3]
            positive = outputs[1::3]
            negative = outputs[2::3]
            assert text.shape == positive.shape == negative.shape, f'text.shape={text.shape}, postive.shape={positive.shape}, negative.shape={negative.shape}'  # NOQA
            positive_inputs = torch.cat((text, positive), dim=0)
            positive_labels = torch.ones_like(positive_inputs[:, :1]).long()
            negative_inputs = torch.cat((text, negative), dim=0)
            negative_labels = torch.zeros_like(negative_inputs[:, :1]).long()
            combined_inputs = torch.cat((positive_inputs, negative_inputs), dim=0)
            combined_labels = torch.cat((positive_labels, negative_labels), dim=0)

            loss = 0.
            if self.w1 > 0:
                loss += self.w1 * cosine_loss(combined_labels, combined_inputs, self.cosine_tau)
            if self.w2 > 0:
                loss += self.w2 * contrastive_with_negative_loss(text, positive, negative, tau=self.ibn_tau)
            if self.w3 > 0:
                loss += self.w3 * angle_loss(combined_labels, combined_inputs, self.angle_tau)
        else:
            raise NotImplementedError
        return loss


class EvaluateCallback(TrainerCallback):
    def __init__(self, model: PreTrainedModel, valid_ds: Dataset, evaluate_fn: Callable, save_dir: Optional[str] = None):
        super().__init__()
        self.model = model
        self.valid_ds = valid_ds
        self.evaluate_fn = evaluate_fn
        self.save_dir = save_dir
        self.best_corrcoef = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        corrcoef, accuracy = self.evaluate_fn(self.valid_ds)
        if corrcoef > self.best_corrcoef:
            self.best_corrcoef = corrcoef
            print('new best corrcoef!')
            if self.save_dir is not None:
                self.model.save_pretrained(self.save_dir)
                print(f'save to {self.save_dir}')
        print(f'corrcoef: {corrcoef}, accuracy: {accuracy}, best corrcoef: {self.best_corrcoef}')


class AnglE:
    cfg_file_name = 'angle.config'
    llm_patterns = [r'.*llama.*', r'.*qwen.*', r'.*baichuan.*']

    def __init__(self,
                 model_name_or_path: str,
                 max_length: int = 512,
                 model_kwargs: Optional[Dict] = None,
                 lora_config_kwargs: Optional[Dict] = None,
                 pooling_strategy: Optional[str] = None,
                 apply_lora: Optional[bool] = None,
                 train_mode: bool = True,
                 load_kbit: Optional[int] = None,
                 is_llm: Optional[bool] = None,
                 pretrained_model_path: Optional[str] = None,
                 pretrained_lora_path: Optional[str] = None,
                 apply_bfloat16: Optional[bool] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None,
                 bellm_class_name: Optional[str] = None,
                 kbit_kwargs: Optional[Dict] = None,
                 **kwargs: Any):
        super().__init__()
        self.max_length = max_length
        self.train_mode = train_mode
        self.pooling_strategy = pooling_strategy
        self.load_kbit = load_kbit
        self.is_llm = is_llm
        if device:
            self.device = device
        else:
            self.device = set_device()
        self.is_bellm = bellm.check_bellm(bellm_class_name)
        if self.is_bellm:
            logger.info('BeLLM detected!')
        if is_llm is None:
            self.is_llm = self.check_llm(model_name_or_path) or self.is_bellm
            if self.is_llm:
                logger.info('LLM detected, automatically set is_llm=True. If it is wrong, you can manually set `is_llm`.')
        self.apply_lora = apply_lora
        if self.apply_lora is None:
            if self.is_llm or self.is_bellm:
                self.apply_lora = True
                logger.info('LLM detected, automatically set apply_lora=True. If it is wrong, you can manually set `apply_lora`.')
        if self.device == 'cuda':
            self.gpu_count = torch.cuda.device_count()
        elif self.device == 'mps':
            self.gpu_count = 1
        else:
            self.gpu_count = 0

        self.prompt = None
        if self.is_llm and not self.is_bellm:
            # do not set prompt for bellm
            logger.info('LLM detected, automatically set prompt. '
                        'You can change this setting by manually configuring the `set_prompt()` function.')
            self.set_prompt()

        self.apply_bfloat16 = apply_bfloat16
        if self.apply_bfloat16 is None and 'llama' in model_name_or_path.lower():
            logger.info('LLaMA detected, automatically set `apply_bfloat16=True`. '
                        'You can change this setting by manually configuring the `apply_bfloat16`.')
            self.apply_bfloat16 = True

        lora_config = None
        if self.apply_lora:
            lora_config = {
                'task_type': TaskType.FEATURE_EXTRACTION,
                'r': 32,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
            }
            if lora_config_kwargs is not None:
                lora_config.update(lora_config_kwargs)
            if train_mode:
                logger.info(f'lora_config={lora_config}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.is_llm and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        kbit_kwargs = kbit_kwargs if kbit_kwargs is not None else {}
        if self.is_llm:
            device_map = "auto"
            MODEL_CLASS = getattr(bellm, bellm_class_name) if self.is_bellm else AutoModelForCausalLM
            if train_mode and self.gpu_count > 1:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            # LLM
            if self.apply_lora:
                lora_config['bias'] = "none"
                lora_config['task_type'] = TaskType.FEATURE_EXTRACTION if self.is_bellm else TaskType.CAUSAL_LM

                if load_kbit == 4:
                    model = MODEL_CLASS.from_pretrained(
                        model_name_or_path,
                        load_in_4bit=True,
                        config=None,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            bnb_4bit_compute_dtype=torch.float32,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type='nf4',
                        ),
                        torch_dtype=torch.float32,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                    if train_mode:
                        model = prepare_model_for_kbit_training(model, **kbit_kwargs)
                    if pretrained_lora_path is not None:
                        print(f'Load lora weight from {pretrained_lora_path}')
                        model = PeftModel.from_pretrained(
                            model,
                            pretrained_lora_path,
                            torch_dtype=torch.float32,
                            device_map=device_map,
                            is_trainable=train_mode
                        )
                    elif train_mode:
                        if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                            target_modules = find_all_linear_names(model, linear_type=bnb.nn.Linear4bit)
                            lora_config['target_modules'] = target_modules
                            logger.info(f'lora target modules={target_modules}')
                        peft_config = LoraConfig(**lora_config)
                        model = get_peft_model(model, peft_config)
                    model = AnglE.kbit_post_handle(model)
                    self.backbone = model
                else:
                    if train_mode:
                        model = MODEL_CLASS.from_pretrained(
                            model_name_or_path,
                            load_in_8bit=load_kbit == 8,
                            torch_dtype=torch.float16 if load_kbit == 16 else torch.float32,
                            device_map=device_map,
                            trust_remote_code=True,
                        )
                        if load_kbit == 8:
                            model = prepare_model_for_int8_training(model, **kbit_kwargs)
                            if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                                target_modules = find_all_linear_names(model)
                                lora_config['target_modules'] = target_modules
                                logger.info(f'lora target modules={target_modules}')
                        if pretrained_lora_path is not None:
                            print(f'Load lora weight from {pretrained_lora_path}')
                            model = PeftModel.from_pretrained(
                                model,
                                pretrained_lora_path,
                                torch_dtype=torch.float16 if load_kbit == 16 else torch.float32,
                                device_map=device_map,
                                is_trainable=train_mode
                            )
                        else:
                            peft_config = LoraConfig(**lora_config)
                            model = get_peft_model(model, peft_config)
                    else:
                        if self.apply_bfloat16:
                            model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                                output_hidden_states=True,
                                                                trust_remote_code=True).bfloat16().to(torch.device(self.device))
                        else:
                            model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                                device_map=device_map,
                                                                output_hidden_states=True,
                                                                trust_remote_code=True,
                                                                load_in_8bit=load_kbit == 8,
                                                                torch_dtype=torch_dtype or torch.float16)
                        if pretrained_lora_path is not None:
                            logger.info(f'Load lora weight from {pretrained_lora_path}')
                            model = PeftModel.from_pretrained(
                                model,
                                pretrained_lora_path,
                                torch_dtype=torch_dtype or torch.float16,
                                device_map=device_map,
                                is_trainable=train_mode
                            )
                    self.backbone = model
            else:
                if self.apply_bfloat16:
                    model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                        output_hidden_states=True,
                                                        trust_remote_code=True).bfloat16().to(torch.device(self.device))
                else:
                    model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                        device_map=device_map,
                                                        output_hidden_states=True,
                                                        trust_remote_code=True,
                                                        load_in_8bit=load_kbit == 8,
                                                        torch_dtype=torch_dtype or torch.float16)
                self.backbone = model
        else:
            # non-LLMs
            if self.apply_lora:
                model = AutoModel.from_pretrained(pretrained_model_path or model_name_or_path, trust_remote_code=True)
                if pretrained_lora_path is not None:
                    model = PeftModel.from_pretrained(
                        model,
                        pretrained_lora_path,
                        is_trainable=train_mode
                    )
                else:
                    if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                        target_modules = find_all_linear_names(model)
                        lora_config['target_modules'] = target_modules
                        logger.info(f'lora target modules={target_modules}')
                    peft_config = LoraConfig(**lora_config)
                    model = get_peft_model(model, peft_config)
                self.backbone = model
            else:
                if pretrained_model_path is not None:
                    logger.info(f'Load pretrained model from {pretrained_model_path}')
                self.backbone = AutoModel.from_pretrained(
                    pretrained_model_path or model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype or "auto")

        if train_mode and self.apply_lora:
            self.backbone.print_trainable_parameters()

        self.backbone.config.use_cache = False
        self.pooler = Pooler(
            self.backbone,
            pooling_strategy=self.pooling_strategy,
            padding_strategy=self.tokenizer.padding_side,
            is_llm=self.is_llm)

        self.__cfg = {
            'model_name_or_path': model_name_or_path,
            'max_length': max_length,
            'model_kwargs': model_kwargs,
            'pooling_strategy': pooling_strategy,
            'lora_config_kwargs': lora_config,
            'apply_lora': apply_lora,
        }
        self.__cfg.update(kwargs)

    def cuda(self):
        if self.load_kbit is None:
            self.backbone = self.backbone.to(torch.device(self.device))
        return self

    def check_llm(self, model_name_or_path: str) -> bool:
        model_name_or_path = model_name_or_path.lower()
        for pattern in AnglE.llm_patterns:
            if re.match(pattern, model_name_or_path):
                return True
        return False

    @staticmethod
    def kbit_post_handle(model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.float32)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    module = module.to(torch.float32)
        return model

    @staticmethod
    def find_pth_path(dirpath: str, config: Dict) -> str:
        if config['save_mode'] == 'best':
            return os.path.join(dirpath, config['best_file_name'])

        pth_list = []
        for fname in os.listdir(dirpath):
            if fname.endswith('.pth'):
                epoch = int(re.search(r'\d+', fname).group())
                pth_list.append((epoch, fname))
        pth_list = sorted(pth_list, key=lambda x: x[0], reverse=True)
        return os.path.join(dirpath, pth_list[0][1])

    @staticmethod
    def from_pretrained(model_name_or_path: str,
                        pretrained_model_path: Optional[str] = None,
                        pretrained_lora_path: Optional[str] = None,
                        is_llm: Optional[bool] = None,
                        pooling_strategy: str = 'cls',
                        train_mode: bool = False,
                        **kwargs):
        angle = AnglE(model_name_or_path,
                      is_llm=is_llm,
                      pretrained_model_path=pretrained_model_path,
                      pretrained_lora_path=pretrained_lora_path,
                      pooling_strategy=pooling_strategy,
                      train_mode=train_mode,
                      **kwargs)
        return angle

    @staticmethod
    def load_config(fpath: str) -> Dict:
        with open(fpath, 'r', encoding='utf-8') as reader:
            return json.load(reader)

    def save_config(self, fpath: str):
        with open(fpath, 'w', encoding='utf-8') as writer:
            json.dump(self.__cfg, writer, ensure_ascii=False, indent=2)

    def detect_dataset_format(self, ds: Dataset):
        for obj in ds:
            return obj['extra']['dataset_format']

    def fit(self,
            train_ds: Dataset,
            valid_ds: Optional[Dataset] = None,
            batch_size: int = 32,
            output_dir: Optional[str] = None,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            warmup_steps: int = 1000,
            logging_steps: int = 10,
            eval_steps: Optional[int] = None,
            save_steps: int = 100,
            save_strategy: str = 'steps',
            save_total_limit: int = 10,
            gradient_accumulation_steps: int = 1,
            fp16: Optional[bool] = None,
            argument_kwargs: Optional[Dict] = None,
            trainer_kwargs: Optional[Dict] = None,
            loss_kwargs: Optional[Dict] = None):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        # save config
        self.save_config(os.path.join(output_dir, AnglE.cfg_file_name))

        if self.gpu_count > 1:
            gradient_accumulation_steps = gradient_accumulation_steps // self.gpu_count
        if fp16 is None and self.is_llm:
            fp16 = True
        else:
            fp16 = False
        if argument_kwargs is None:
            argument_kwargs = {}
        if trainer_kwargs is None:
            trainer_kwargs = {}
        callbacks = None
        if valid_ds is not None:
            best_ckpt_dir = None
            if output_dir is not None:
                best_ckpt_dir = os.path.join(output_dir, 'best-checkpoint')
            evaluate_callback = EvaluateCallback(self.backbone, valid_ds,
                                                 partial(self.evaluate, batch_size=batch_size, device=self.device),
                                                 save_dir=best_ckpt_dir)
            callbacks = [evaluate_callback]
        trainer = AngleTrainer(
            pooler=self.pooler,
            model=self.backbone,
            dataset_format=self.detect_dataset_format(train_ds),
            train_dataset=train_ds,
            eval_dataset=None,
            loss_kwargs=loss_kwargs,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                fp16=fp16,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                eval_steps=eval_steps,
                save_steps=save_steps,
                output_dir=output_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False if self.gpu_count > 1 else None,
                label_names=['labels', 'seperate_ids', 'extra'],
                **argument_kwargs,
            ),
            callbacks=callbacks,
            data_collator=AngleDataCollator(
                self.tokenizer, return_tensors="pt", max_length=self.max_length
            ),
            **trainer_kwargs
        )
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.backbone = torch.compile(self.backbone)

        trainer.train()
        self.backbone.save_pretrained(output_dir)

    def evaluate(self, data: Dataset, batch_size: int = 32, threshold: Optional[float] = None, device: Any = None):
        self.backbone.eval()
        data_collator = AngleDataCollator(
            self.tokenizer,
            return_tensors="pt",
            max_length=self.max_length,
        )
        y_trues, y_preds = [], []
        # for X, y in data.make_iter(random=False):
        for features in tqdm(chunked_iter(data, batch_size), desc='Evaluate'):
            X = data_collator(features)
            y = X.pop('labels', None)
            y_trues.extend(y[::2, 0].detach().cpu().numpy())
            with torch.no_grad():
                X.to(device or self.device)
                x_vecs = self.pooler(X).detach().float().cpu().numpy()
            x_vecs = l2_normalize(x_vecs)
            pred = (x_vecs[::2] * x_vecs[1::2]).sum(1)
            y_preds.extend(pred)

        y_trues, y_preds = np.array(y_trues), np.array(y_preds)
        corrcoef = compute_corrcoef(y_trues, y_preds)
        if threshold is None:
            _, accuracy = optimal_threshold(y_trues, y_preds)
        else:
            accuracy = np.mean((y_trues > 0.5) == (y_preds > threshold))
        return corrcoef, accuracy

    def set_prompt(self, prompt: str = Prompts.A):
        self.prompt = prompt
        if self.prompt is not None:
            logger.info('Prompt is set, the prompt will be automatically applied during the encoding phase. '
                        'To disable prompt setting, please configure set_prompt(prompt=None)')

    def encode(self,
               inputs: Union[List[str], Tuple[str], List[Dict], str],
               max_length: Optional[int] = None,
               end_with_eos: bool = False,
               to_numpy: bool = True,
               device: Any = None):
        if device is None:
            device = self.device
        self.backbone.eval()
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if self.prompt is not None:
            for i, obj in enumerate(inputs):
                assert isinstance(obj, dict), 'The prompt has been set, please pass a dict like {"prompt_key": "text"}'
                inputs[i] = self.prompt.format(**obj)
        max_length = max_length or self.max_length
        if end_with_eos:
            max_length -= 1

        if end_with_eos:
            tok = self.tokenizer(
                inputs,
                padding=False,
                return_attention_mask=False,
                max_length=max_length or self.max_length,
                truncation=True)
            tok['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in tok['input_ids']]
            tok = self.tokenizer.pad(tok, padding=True, return_attention_mask=True, return_tensors='pt')
        else:
            tok = self.tokenizer(
                inputs,
                padding='longest',
                max_length=max_length or self.max_length,
                truncation=True,
                return_tensors='pt')
        tok.to(device)
        with torch.no_grad():
            output = self.pooler(tok)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output

    def export_onnx(self):
        pass
