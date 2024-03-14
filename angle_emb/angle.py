# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import copy
import random
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

from .utils import logger


DEFAULT_LLM_PATTERNS = [r'.*llama.*', r'.*qwen.*', r'.*baichuan.*', r'.*mistral.*']


def set_device() -> str:
    """
    Set device automatically

    :return: str, device name
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def find_all_linear_names(model: PreTrainedModel, linear_type: Optional[object] = None) -> List[str]:
    """
    Find all linear layer names

    :param model: PreTrainedModel
    :param linear_type: Optional[object] = None, linear type, such as nn.Linear and bnb.nn.Linear4bit.

    :return: List[str], linear layer names
    """
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


def categorical_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """
    Compute categorical crossentropy

    :param y_true: torch.Tensor, ground truth
    :param y_pred: torch.Tensor, model output
    :param from_logits: bool, `True` means y_pred has not transformed by softmax, default True

    :return: torch.Tensor, loss value
    """
    if from_logits:
        return -(F.log_softmax(y_pred, dim=1) * y_true).sum(dim=1)
    return -(torch.log(y_pred, dim=1) * y_true).sum(dim=1)


def cosine_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """
    Compute cosine loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 20

    :return: torch.Tensor, loss value
    """  # NOQA
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
    """
    Compute angle loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 1.0

    :return: torch.Tensor, loss value
    """  # NOQA
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
                           negative_weights: float = 0.0) -> torch.Tensor:
    """
    Compute in-batch negative loss, i.e., contrastive loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [x[0][0], x[0][1], x[1][0], x[1][1], ...], where (x[0][0], x[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 20.0
    :param negative_weights: float, negative weights, default 0.0

    :return: torch.Tensor, loss value
    """  # NOQA
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


def contrastive_with_negative_loss(
        text: torch.Tensor,
        pos: torch.Tensor,
        neg: Optional[torch.Tensor] = None,
        tau: float = 20.0) -> torch.Tensor:
    """
    Compute contrastive with negative loss

    :param text: torch.Tensor, text.
    :param pos: torch.Tensor, positive samples of text.
    :param neg: torch.Tensor, negative samples of text.
    :param tau: float, scale factor, default 20.0

    :return: torch.Tensor, loss value
    """
    target = torch.cat((pos, neg), dim=0) if neg is not None else pos  # (2B, D)
    q_norm = torch.nn.functional.normalize(text, p=2, dim=1)  # (B, D)
    t_norm = torch.nn.functional.normalize(target, p=2, dim=1)  # (2B, D)
    scores = torch.mm(q_norm, t_norm.transpose(0, 1)) * tau  # (B, 2B)
    labels = torch.tensor(
        range(len(scores)), dtype=torch.long, device=scores.device
    )
    return nn.CrossEntropyLoss()(scores, labels)


def compute_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute correlation coefficients

    :param x: np.ndarry, x array
    :param y: np.ndarry, y array

    :return: float
    """
    return scipy.stats.spearmanr(x, y).correlation


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array using L2

    :param arr: np.ndarray, input array
    :return: np.ndarray
    """
    norms = (arr**2).sum(axis=1, keepdims=True)**0.5
    return arr / np.clip(norms, 1e-8, np.inf)


def optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute optimal threshold

    :param y_true: np.ndarray, y_true
    :param y_pred: np.ndarray, y_true

    :return: Tuple[float, float]
    """
    loss = lambda t: -np.mean((y_true > 0.5) == (y_pred > np.tanh(t)))  # NOQA
    result = scipy.optimize.minimize(loss, 1, method='Powell')
    return np.tanh(result.x), -result.fun


def check_llm(model_name_or_path: str, llm_regex_patterns: List[str] = None) -> bool:
    if llm_regex_patterns is not None:
        llm_regex_patterns += DEFAULT_LLM_PATTERNS
    else:
        llm_regex_patterns = DEFAULT_LLM_PATTERNS
    model_name_or_path = model_name_or_path.lower()
    for pattern in llm_regex_patterns:
        if re.match(pattern, model_name_or_path):
            return True
    return False


def get_pooling(outputs: torch.Tensor,
                inputs: Dict,
                pooling_strategy: str,
                padding_strategy: str = 'right') -> torch.Tensor:
    """
    get pooling

    :param outputs:  torch.Tensor. Model outputs (without pooling)
    :param inputs:  Dict. Model inputs
    :param pooling_strategy:  str. Pooling strategy ['cls', 'cls_avg', 'cls_max', 'last', 'avg', 'max', 'all', index]
    :param padding_strategy:  str. Padding strategy of tokenizers (`left` or `right`).
        It can be obtained by `tokenizer.padding_side`.
    """
    if pooling_strategy == 'cls':
        outputs = outputs[:, 0]
    elif pooling_strategy == 'cls_avg':
        avg = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        outputs = (outputs[:, 0] + avg) / 2.0
    elif pooling_strategy == 'cls_max':
        maximum, _ = torch.max(outputs * inputs["attention_mask"][:, :, None], dim=1)
        outputs = (outputs[:, 0] + maximum) / 2.0
    elif pooling_strategy == 'last':
        batch_size = inputs['input_ids'].shape[0]
        sequence_lengths = -1 if padding_strategy == 'left' else inputs["attention_mask"].sum(dim=1) - 1
        outputs = outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
    elif pooling_strategy == 'avg':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
    elif pooling_strategy == 'max':
        outputs, _ = torch.max(outputs * inputs["attention_mask"][:, :, None], dim=1)
    elif pooling_strategy == 'all':
        # keep outputs
        pass
    elif isinstance(pooling_strategy, int) or pooling_strategy.isnumeric():
        # index
        outputs = outputs[:, int(pooling_strategy)]
    else:
        raise NotImplementedError(
            'please specify pooling_strategy from [`cls`, `last`, `avg`, `max`, `last_avg`, `all`, int]')
    return outputs


def get_geometric_hidden_sizes(base: int = 8, max_hidden: int = 768) -> List[int]:
    """
    get geometric hidden size series list given a hidden size range

    """
    lst = []
    s = base
    while s < max_hidden:
        lst.append(s)
        s *= 2
    return lst


class Prompts:
    """
    Predefined prompts. Follow the model usage to choose the corresponding prompt.

    Example::

            from angle_emb import Prompts

            # list all pre-defined prompts
            print(Prompts.list_prompts())
            # set prompt
            angle.set_prompt(prompt=Prompts.A)

    """

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
    """
    Predefined Data Formats.

    Check all available formats:

            from angle_emb import DatasetFormats

            print(DatasetFormats.list_formats())

    """

    """
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
    """
    A = 'text1,text2,label'

    """
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
    """
    B = 'text,positive,negative'

    """
    format C: text,positive
    input format: [
        text[0],
        positive[0],
        text[1],
        positive[1],
        ...
    ]
    """
    C = 'text,positive'

    @classmethod
    def list_formats(cls):
        for key, val in DatasetFormats.__dict__.items():
            if key.startswith('_') or key == 'list_formats':
                continue
            print(f'DatasetFormats.{key}', '=', f"'{val}'")


class AngleDataTokenizer:
    """
    Tokenize data using AngleDataTokenizer.

    :param tokenizer: PreTrainedTokenizerBase. Tokenizer
    :param max_length: Optional[int]. Specify max length
    :param prompt_template: Optional[str], set prompt template, it will be applied to all input texts. Default None
    :param extra_columns: Optional[List[str]].
        If providing multiple placeholders in prompt_template, specify their name via extra_columns. Default None
    :param dataset_format: Optional[str]. Specify dataset_format from DatasetFormats. Default None.
        It will automatically detect the dataset format.
    :param end_with_eos: bool. Specify whether ends with the eos token. Default False.

    Example::

            from datasets import load_dataset
            from angle_emb import AnglE, AngleDataTokenizer

            # define dataset
            ds = load_dataset('your_dataset')
            # define angle
            angle = AnglE(*args, **kwargs)
            # tokenize data
            train_ds = ds['train'].shuffle().map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
            valid_ds = ds['validation'].map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)

    """
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
            elif 'text' in data and 'positive' in data and 'negative' not in data and 'label' not in data:
                self.dataset_format = DatasetFormats.C
                logger.info(f'Detect DatasetFormats.C: {DatasetFormats.C}')
            else:
                raise NotImplementedError('Currently only support two dataset formats'
                                          'DatasetFormats A: must include three columns: `text1`, `text2`, and `label`.'
                                          'DatasetFormats B: mut include three columns: `text`, `positive`, `negative`'
                                          'DatasetFormats C: mut include three columns: `text`, `positive`')
        text_columns = None
        if self.dataset_format == DatasetFormats.A:
            text_columns = ['text1', 'text2']
        elif self.dataset_format == DatasetFormats.B:
            text_columns = ['text', 'positive', 'negative']
        elif self.dataset_format == DatasetFormats.C:
            text_columns = ['text', 'positive']

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
            max_length = self.max_length - len(self.prompt_template_tok['input_ids']) - extra_length
            for text_column in text_columns:
                tok = self.tokenizer(data[text_column],
                                     max_length=max_length,
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
                    logger.info(f"data data: token ids={tok['input_ids']}, prompt_token_ids={self.prompt_template_tok['input_ids']}")  # NOQA
                    tok['input_ids'] = self.fix_bad_data(tok['input_ids'], self.prompt_template_tok['input_ids'])
                    try:
                        assert len(tok['input_ids']) == len(tok['attention_mask'])
                        assert tok['input_ids'][-1] == self.prompt_template_tok['input_ids'][-1]
                        logger.info('fixed it ;)')
                        logger.info(f"new data, token ids={tok['input_ids']}, prompt_token_ids={self.prompt_template_tok['input_ids']}")  # NOQA
                    except AssertionError:
                        logger.info('failed to fix it :( skip it...')

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
    """
    AngleDataCollator. It will be implicitly used in AnglE.fit().
    It can only handle the tokenized data using AngleDataTokenizer.

    :param tokenizer:  PreTrainedTokenizerBase
    :param padding:   Union[bool, str, PaddingStrategy], padding strategy
    :param max_length:  Optional[int], max length
    :param return_tensors:  str
    :param filter_duplicate: bool. Whether filter duplicate data
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_length: Optional[int] = None
    return_tensors: str = "pt"
    filter_duplicate: bool = True

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        has_token_type_ids = "token_type_ids" in features[0]
        end_with_eos = features[0]['extra']['end_with_eos']

        new_features = []
        duplicate_set = set()
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
            current_features = []
            is_duplicate = False
            for seperate_id in range(1, max_seperate_id + 1):
                start_idx = seperate_ids.index(seperate_id)
                new_feature = {}
                new_input_ids = input_ids[prev_start_idx:start_idx]
                if tuple(new_input_ids) in duplicate_set:
                    is_duplicate = True
                    if self.filter_duplicate:
                        break
                duplicate_set.add(tuple(new_input_ids))
                new_feature['input_ids'] = new_input_ids
                new_feature['attention_mask'] = attention_mask[prev_start_idx:start_idx]
                if has_token_type_ids:
                    new_feature['token_type_ids'] = token_type_ids[prev_start_idx:start_idx]
                new_feature['labels'] = feature['labels']
                current_features.append(new_feature)
                prev_start_idx = start_idx

            # last
            new_feature = {}
            new_input_ids = input_ids[prev_start_idx:]
            if tuple(new_input_ids) in duplicate_set:
                is_duplicate = True
            duplicate_set.add(tuple(new_input_ids))
            new_feature['input_ids'] = new_input_ids
            new_feature['attention_mask'] = attention_mask[prev_start_idx:]
            if has_token_type_ids:
                new_feature['token_type_ids'] = token_type_ids[prev_start_idx:]
            new_feature['labels'] = feature['labels']
            current_features.append(new_feature)

            if self.filter_duplicate and is_duplicate:
                continue
            new_features += current_features

        # remove features
        del features

        if end_with_eos:
            features = {}
            features['input_ids'] = [feature['input_ids'] + [self.tokenizer.eos_token_id] for feature in new_features]
            features = self.tokenizer.pad(
                features,
                padding=self.padding,
                return_attention_mask=True,
                return_tensors=return_tensors)
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
    """
    Using Pooler to obtain sentence embeddings.

    :param model: PreTrainedModel
    :param pooling_strategy: Optional[str]. Currently support [`cls`, `last`, `avg`, `cls_avg`, `max`]. Default None.
    :param padding_strategy: Optional[str]. `left` or `right`. Default None.
    :param is_llm: bool. Default False
    """
    def __init__(self,
                 model: PreTrainedModel,
                 pooling_strategy: Optional[Union[int, str]] = None,
                 padding_strategy: Optional[str] = None,
                 is_llm: bool = False):
        self.model = model
        self.pooling_strategy = pooling_strategy
        self.padding_strategy = padding_strategy
        self.is_llm = is_llm

    def __call__(self, inputs: Dict, layer_index: int = -1, embedding_size: Optional[int] = None,
                 return_all_layer_outputs: bool = False) -> torch.Tensor:
        """
        :param inputs: Dict. Model inputs.
        :param layer_index: int. Get embeddings from specific layer.
        :param embedding_size: int. Set embedding size for sentence embeddings for 2DMSE models.
        """
        all_layer_outputs = self.model(output_hidden_states=True, return_dict=True, **inputs).hidden_states
        if return_all_layer_outputs:
            return all_layer_outputs
        outputs = all_layer_outputs[layer_index]
        if self.is_llm:
            batch_size = inputs['input_ids'].shape[0]
            sequence_lengths = -1 if self.padding_strategy == 'left' else inputs["attention_mask"].sum(dim=1) - 1
            outputs = outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
        else:
            outputs = get_pooling(outputs, inputs, self.pooling_strategy, padding_strategy=self.padding_strategy)
        if embedding_size is not None:
            # topk embedding size
            return outputs[:, :embedding_size]
        return outputs


class AngleTrainer(Trainer):
    """
    Custom Huggingface Trainer for AnglE.

    :param pooler: Pooler. Required
    :param loss_kwargs: Optional[Dict]. Default None.
    :param dataset_format: str. Default DatasetFormats.A
    :param fixed_teacher_name_or_path: Optional[str]. For distribution alignment.
    :param **kwargs: other parameters of Trainer.
    """
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = DatasetFormats.A,
                 fixed_teacher_name_or_path: Optional[str] = None,
                 alignment_pooling_strategy: str = 'cls',
                 **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_fct = AngleLoss(dataset_format=dataset_format, **loss_kwargs)
        self.fixed_teacher_name_or_path = fixed_teacher_name_or_path
        self.alignment_pooling_strategy = alignment_pooling_strategy
        if fixed_teacher_name_or_path is not None:
            assert not check_llm(fixed_teacher_name_or_path), ('Currently not support LLMs alignment,'
                                                               f' teacher={fixed_teacher_name_or_path}')
            assert self.pooler.pooling_strategy == 'all', ('fixed_teacher_name_or_path detected!'
                                                           ' please set --pooling_strategy all')
            fixed_teacher_backbone = AutoModel.from_pretrained(
                fixed_teacher_name_or_path,
                trust_remote_code=True,
                torch_dtype="auto")

            fixed_teacher_backbone.config.use_cache = False
            self.fixed_teacher_pooler = Pooler(
                fixed_teacher_backbone,
                pooling_strategy='all',
                padding_strategy=self.pooler.padding_strategy,
                is_llm=False)
            self.kl_loss_fct = nn.KLDivLoss(reduction='batchmean')
            logger.info(f'Train with alignment, teacher={fixed_teacher_name_or_path}')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        if self.fixed_teacher_name_or_path is not None:
            all_outputs = self.pooler(inputs)
            outputs = get_pooling(all_outputs, inputs,
                                  self.alignment_pooling_strategy,
                                  self.pooler.padding_strategy)
            loss = self.loss_fct(labels, outputs)
            with torch.no_grad():
                self.fixed_teacher_pooler.model = self.fixed_teacher_pooler.model.to(self.pooler.model.device)
                all_fixed_outputs = self.fixed_teacher_pooler(inputs)

            alignment_loss = self.kl_loss_fct(
                F.log_softmax(all_outputs, dim=-1),
                F.softmax(all_fixed_outputs, dim=-1)
            )
            loss += alignment_loss
        else:
            outputs = self.pooler(inputs)
            loss = self.loss_fct(labels, outputs)

        return (loss, outputs) if return_outputs else loss


class AngleTDMSETrainer(AngleTrainer):
    """
    Custom Huggingface Trainer for AnglE 2DMSE.

    :param pooler: Pooler. Required
    :param loss_kwargs: Optional[Dict]. Default None.
    :param dataset_format: str. Default DatasetFormats.A
    :param fixed_teacher_name_or_path: Optional[str]. For distribution alignment.

    :param **kwargs: other parameters of Trainer.
    """
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = DatasetFormats.A,
                 fixed_teacher_name_or_path: Optional[str] = None,
                 tdmse_kl_temperature: float = 1.0,
                 tdmse_teacher_lambda: float = 1.0,
                 tdmse_student_lambda: float = 1.0,
                 apply_tdmse_kl: bool = True,
                 **kwargs):
        super().__init__(pooler=pooler,
                         loss_kwargs=loss_kwargs,
                         dataset_format=dataset_format,
                         fixed_teacher_name_or_path=fixed_teacher_name_or_path,
                         **kwargs)
        self.tdmse_kl_temperature = tdmse_kl_temperature
        self.tdmse_teacher_lambda = tdmse_teacher_lambda
        self.tdmse_student_lambda = tdmse_student_lambda
        self.apply_tdmse_kl = apply_tdmse_kl
        self.n_layers = self.pooler.model.config.num_hidden_layers
        self.hidden_size = self.pooler.model.config.hidden_size
        self.tdmse_hidden_sizes = get_geometric_hidden_sizes(base=8, max_hidden=self.hidden_size)
        self.kl_loss_fct = nn.KLDivLoss(reduction='batchmean')
        logger.info('Train with 2DMSE!')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        # layer
        sample_layer = random.randint(1, self.n_layers - 1)
        pooling_strategy = (self.alignment_pooling_strategy
                            if self.pooler.pooling_strategy == 'all'
                            else self.pooler.pooling_strategy)
        all_layer_outputs = self.pooler(inputs, layer_index=-1, return_all_layer_outputs=True)
        all_teacher_outputs = all_layer_outputs[-1]
        teacher_outputs = get_pooling(all_teacher_outputs, inputs,
                                      pooling_strategy,
                                      self.pooler.padding_strategy)
        all_student_outputs = all_layer_outputs[sample_layer]
        student_outputs = get_pooling(all_student_outputs,
                                      inputs,
                                      pooling_strategy,
                                      self.pooler.padding_strategy)

        teacher_kl_outputs = teacher_outputs
        if self.fixed_teacher_name_or_path is not None:
            with torch.no_grad():
                self.fixed_teacher_pooler.model = self.fixed_teacher_pooler.model.to(self.pooler.model.device)
                all_fixed_outputs = self.fixed_teacher_pooler(inputs)
                teacher_kl_outputs = get_pooling(all_fixed_outputs,
                                                 inputs,
                                                 self.alignment_pooling_strategy,
                                                 self.pooler.padding_strategy)

        teacher_loss = self.loss_fct(labels, teacher_outputs)
        loss1 = teacher_loss
        student_loss = self.loss_fct(labels, student_outputs)
        loss1 += student_loss / sample_layer
        if self.apply_tdmse_kl and self.tdmse_student_lambda > 0:
            kl_loss = self.kl_loss_fct(
                F.log_softmax(student_outputs / self.tdmse_kl_temperature, dim=-1),
                F.softmax(teacher_kl_outputs / self.tdmse_kl_temperature, dim=-1)
            ) * self.tdmse_kl_temperature
            loss1 += kl_loss

        # feature
        hidden_size = random.choice(self.tdmse_hidden_sizes)
        slimmed_teacher_outputs = teacher_outputs[:, :hidden_size]
        slimmed_student_outputs = student_outputs[:, :hidden_size]

        slimmed_teacher_loss = self.loss_fct(labels, slimmed_teacher_outputs)
        loss2 = slimmed_teacher_loss
        slimmed_student_loss = self.loss_fct(labels, slimmed_student_outputs)
        loss2 += slimmed_student_loss / sample_layer

        loss = loss1 + loss2

        if self.fixed_teacher_name_or_path is not None:
            alignment_loss = self.kl_loss_fct(
                F.log_softmax(all_teacher_outputs, dim=-1),
                F.softmax(all_fixed_outputs, dim=-1)
            )
            loss += alignment_loss
        return (loss, teacher_outputs) if return_outputs else loss


class AngleLoss:
    """
    Configure AngleLoss.

    :param w1: float. weight for cosine_loss. Default 1.0
    :param w2: float. weight for contrastive loss. Default 1.0
    :param w3: float. weight for angle loss. Default 1.0
    :param cosine_tau: float. tau for cosine loss. Default 20.0
    :param ibn_tau: float. tau for contrastive loss. Default 20.0
    :param angle_tau: float. tau for angle loss. Default 1.0
    :param dataset_format: Optional[str]. Default None.
    """
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

            _, fea_dim = text.shape
            positive_inputs = torch.stack((text, positive), dim=1).reshape(-1, fea_dim)  # zip(text, positive)
            positive_labels = torch.ones_like(positive_inputs[:, :1]).long()
            negative_inputs = torch.stack((text, negative), dim=1).reshape(-1, fea_dim)  # zip(text, negative)
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
        elif self.dataset_format == DatasetFormats.C:
            text = outputs[::2]
            positive = outputs[1::2]
            loss = contrastive_with_negative_loss(text, positive, neg=None, tau=self.ibn_tau)
        else:
            raise NotImplementedError
        return loss


class EvaluateCallback(TrainerCallback):
    """
    Custom TrainerCallback for Angle.
    This callback will compute corrcoef for each epoch.

    :param model: PreTrainedModel.
    :param valid_ds: Dataset.
    :param evaluate_fn: Callable. It will receive valid_ds as input like `evaluate_fn(valid_ds)`.
    :param save_dir: Optional[str]. specify dir to save model with best results.
    """
    def __init__(self,
                 model: PreTrainedModel,
                 valid_ds: Dataset,
                 evaluate_fn: Callable,
                 save_dir: Optional[str] = None):
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
    """
    AnglE. Everything is here👋

    :param model_name_or_path: str, model name or path.
    :param max_length: int. Default 512
    :param model_kwargs: Optional[Dict]. kwargs for model.
    :param lora_config_kwargs: Optional[Dict]. kwargs for peft lora_config.
        details refer to: https://huggingface.co/docs/peft/tutorial/peft_model_config
    :param pooling_strategy: Optional[str]. Pooling strategy.
        Currently support [`cls`, `last`, `avg`, `cls_avg`, `max`]
    :param apply_lora: Optional[bool]. Whether apply lora. Default None.
    :param train_mode: bool. Whether load for training. Default True.
    :param load_kbit: Optional[int]. Specify kbit training from [4, 8, 16]. Default None.
    :param is_llm: Optional[bool]. Whether the model is llm. Default None.
    :param pretrained_model_path: Optional[str]. Default None.
    :param pretrained_lora_path: Optional[str]. Default None.
    :param apply_bfloat16: Optional[bool]. Whether load using torch.bfloat16. Default None.
    :param torch_dtype: Optional[torch.dtype]. Specify torch_dtype. Default None.
    :param device: Optional[str]. Specify device. Default None.
    :param kbit_kwargs: Optional[Dict]. kwargs for kbit. Default None.
        details refer to: https://huggingface.co/docs/peft/package_reference/peft_model#peft.prepare_model_for_kbit_training
    :param **kwargs: Any.
    """  # NOQA
    cfg_file_name = 'angle.config'

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
        if is_llm is None:
            self.is_llm = check_llm(model_name_or_path)
            if self.is_llm:
                logger.info('LLM detected, automatically set is_llm=True.'
                            'If it is wrong, you can manually set `is_llm`.')
        self.apply_lora = apply_lora
        if self.apply_lora is None:
            if self.is_llm:
                self.apply_lora = True
                logger.info('LLM detected, automatically set apply_lora=True.'
                            'If it is wrong, you can manually set `apply_lora`.')
        if self.device == 'cuda':
            self.gpu_count = torch.cuda.device_count()
        elif self.device == 'mps':
            self.gpu_count = 1
        else:
            self.gpu_count = 0

        self.prompt = None
        if self.is_llm:
            logger.info('LLM detected, automatically set prompt. '
                        'You can change this setting by manually configuring the `set_prompt()` function.')
            self.set_prompt()

        self.apply_bfloat16 = apply_bfloat16
        if self.apply_bfloat16 is None and 'llama' in model_name_or_path.lower():
            logger.info('LLaMA detected, automatically set `apply_bfloat16=True`. '
                        'You can change this setting by manually configuring the `apply_bfloat16`.')
            self.apply_bfloat16 = True

        if torch_dtype is None:
            torch_dtype = torch.float32 if train_mode else None

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
            MODEL_CLASS = AutoModelForCausalLM
            if train_mode and self.gpu_count > 1:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            # LLM
            if self.apply_lora:
                lora_config['bias'] = "none"
                lora_config['task_type'] = TaskType.CAUSAL_LM

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
                                                                trust_remote_code=True).bfloat16()
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
                                                        trust_remote_code=True).bfloat16()
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

        # full_backbone is used to 2DMSE inference
        self.full_backbone = None
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

    def to(self, device: Any):
        if isinstance(device, str):
            device = torch.device(device)
        self.backbone = self.backbone.to(device)
        self.device = device
        return self

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
        """
        Load AnglE from pretrained model.

        :param model_name_or_path: str, model name or path. Required.
        :param pretrained_model_path: Optional[str].
        :param pretrained_lora_path: Optional[str].
        :param is_llm: Optional[bool].
        :param pooling_strategy: str. Pooling Strategy. Default `cls`.
        :param train_mode: bool. Default False.
        :param kwargs: Other kwargs for AnglE.

        :return: AnglE object.

        Example::

                from angle_emb import AnglE

                angle = AnglE.from_pretrained(model_name_or_path)
                # fit
                angle.fit(*args, **kwargs)
                # inference
                angle.encode(*args, **kwargs)
        """
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
            loss_kwargs: Optional[Dict] = None,
            apply_tdmse: bool = False,
            filter_duplicate: bool = True):
        """
        Fit using AnglE.

        :param train_ds: Dataset. tokenized train dataset. Required.
        :param valid_ds: Optional[Dataset]. tokenized valid dataset. Default None.
        :param batch_size: int. Default 32.
        :param output_dir: Optional[str]. save dir. Default None.
        :param epochs: int. Default 1.
        :param learning_rate: float. Default 1e-5.
        :param warmup_steps: int. Default 1000.
        :param logging_steps: int. Default 10.
        :param eval_steps: Optional[int]. Default None.
        :param save_steps: int. Default 100.
        :param save_strategy: str. Default steps.
        :param save_total_limit: int. Default 10.
        :param gradient_accumulation_steps: int. Default 1.
        :param fp16: Optional[bool]. Default None.
        :param argument_kwargs: Optional[Dict]. kwargs for TrainingArguments.
            refer to: https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments
        :param trainer_kwargs: Optional[Dict]. kwargs for AngleTrainer.
        :param loss_kwargs: Optional[Dict]. kwargs for AngleLoss.
        :param apply_tdmse: bool, whether apply TDMSE training.
        """  # NOQA
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

        CustomTrainer = AngleTDMSETrainer if apply_tdmse else AngleTrainer
        trainer = CustomTrainer(
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
                self.tokenizer, return_tensors="pt", max_length=self.max_length, filter_duplicate=filter_duplicate
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
            filter_duplicate=False,
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
               layer_index: int = -1,
               embedding_size: Optional[int] = None,
               device: Optional[Any] = None):
        """
        encode texts.

        :param inputs: Union[List[str], Tuple[str], List[Dict], str]. Input texts. Required.
        :param max_length: Optional[int]. Default None.
        :param to_numpy: bool. Default True.
        :param layer_index: int. Obtain specific layer's sentence embeddings (for 2DMSE).
        :param embedding_size: Optional[int]. Specify embedding size (for 2DMSE).
        :param device: Optional[Any]. Default None.
        """
        if layer_index != -1 and self.full_backbone is None:
            self.full_backbone = copy.deepcopy(self.backbone)

        if layer_index != -1:
            self.backbone.encoder.layer = self.full_backbone.encoder.layer[:layer_index]

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
            output = self.pooler(tok, layer_index=layer_index, embedding_size=embedding_size)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output

    def export_onnx(self):
        pass
