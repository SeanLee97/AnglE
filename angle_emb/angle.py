# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import math
import random
from functools import partial
from typing import Any, Dict, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoModel, AutoModelForMaskedLM, AutoTokenizer,
    PreTrainedModel, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from huggingface_hub import repo_exists
from peft import (
    get_peft_model, LoraConfig, TaskType, PeftModel,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer

from .base import AngleBase
from .utils import logger
from .evaluation import CorrelationEvaluator
from .version import __version__


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


def angle_loss(y_true: torch.Tensor, y_pred: torch.Tensor, tau: float = 1.0, pooling_strategy: str = 'sum'):
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
    if pooling_strategy == 'sum':
        pooling = torch.sum(y_pred, dim=1)
    elif pooling_strategy == 'mean':
        pooling = torch.mean(y_pred, dim=1)
    else:
        raise ValueError(f'Unsupported pooling strategy: {pooling_strategy}')
    y_pred = torch.abs(pooling) * tau  # absolute delta angle
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
                padding_side: str) -> torch.Tensor:
    """ Pooling the model outputs.

    :param outputs:  torch.Tensor. Model outputs (without pooling)
    :param inputs:  Dict. Model inputs
    :param pooling_strategy:  str.
        Pooling strategy [`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]
    :param padding_side:  str. Padding strategy of tokenizers (`left` or `right`).
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
        sequence_lengths = -1 if padding_side == 'left' else inputs["attention_mask"].sum(dim=1) - 1
        outputs = outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths]
    elif pooling_strategy in ['avg', 'mean']:
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(1)
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
            'please specify pooling_strategy from '
            '[`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]')
    return outputs


class Prompts:
    """
    Predefined prompts. Follow the model usage to choose the corresponding prompt.

    Example::

            from angle_emb import Prompts

            # list all pre-defined prompts
            print(Prompts.list_prompts())
            # set prompt
            angle.encode(*, prompt=Prompts.A)
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
    :param fix_data: bool. Specify whether fix the data. Only works when prompt_template is not None. Default True.

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
                 end_with_eos: bool = False,
                 fix_data: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.prompt_template_tok = None
        self.extra_columns = extra_columns
        self.dataset_format = dataset_format
        self.end_with_eos = end_with_eos
        self.fix_data = fix_data
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

        if self.prompt_template_tok is not None and self.fix_data:
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
            'end_with_eos': self.end_with_eos,
            'prompt_token_ids': self.prompt_template_tok['input_ids'] if self.prompt_template_tok is not None else None,
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
    :param coword_random_mask_rate: float. Default 0.0.
        If set it greater than 0, the random maked token prediction will be added to the training loss.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_length: Optional[int] = None
    return_tensors: str = "pt"
    filter_duplicate: bool = True
    coword_random_mask_rate: float = 0.0
    special_token_id_names: List[str] = field(default_factory=lambda: [
        'bos_token_id', 'eos_token_id', 'unk_token_id', 'sep_token_id',
        'pad_token_id', 'cls_token_id', 'mask_token_id'])

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """ Collate function for AngleDataTokenizer.

        :param features: List[Dict]. Tokenized data
        :param return_tensors: str. Default "pt"
        :return: Dict[str, torch.Tensor]. Collated data
        """
        if return_tensors is None:
            return_tensors = self.return_tensors
        has_token_type_ids = "token_type_ids" in features[0]
        end_with_eos = features[0]['extra']['end_with_eos']
        prompt_token_ids = set(features[0]['extra']['prompt_token_ids'] or [])
        special_token_ids = set()
        for name in self.special_token_id_names:
            if hasattr(self.tokenizer, name):
                special_token_ids.add(getattr(self.tokenizer, name))
        for token in self.tokenizer.additional_special_tokens:
            special_token_ids.add(self.tokenizer.encode(token)[0])
        predefined_token_ids = prompt_token_ids | special_token_ids

        if self.coword_random_mask_rate > 0:
            if not isinstance(self.tokenizer.mask_token_id, int):
                raise NotImplementedError("Failed to train with random mask common tokens"
                                          "since the tokenizer does not have a mask token."
                                          "Please set a special mask token to the tokenizer manually.")

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

            if self.coword_random_mask_rate > 0:
                first_text_tokens = set(current_features[0]['input_ids']) - predefined_token_ids
                common_tokens_with_first_text = set()
                for cnt_fea in current_features[1:]:
                    cnt_tokens = set(cnt_fea['input_ids']) - predefined_token_ids
                    cnt_common_tokens = list(cnt_tokens & first_text_tokens)
                    common_tokens_with_first_text.update(cnt_common_tokens)
                    if cnt_common_tokens:
                        sample_size = max(1, int(len(cnt_common_tokens) * self.coword_random_mask_rate))
                        sampled_mask_tokens = random.sample(cnt_common_tokens, sample_size)
                        cnt_fea['input_ids'] = [self.tokenizer.mask_token_id
                                                if idx in sampled_mask_tokens
                                                else idx for idx in cnt_fea['input_ids']]
                    cnt_fea['mask_target_labels'] = cnt_fea['input_ids']

                # mask first text
                common_tokens_with_first_text = list(common_tokens_with_first_text)
                if common_tokens_with_first_text:
                    sample_size = max(1, int(len(common_tokens_with_first_text) * self.coword_random_mask_rate))
                    sampled_mask_tokens = random.sample(common_tokens_with_first_text, sample_size)
                    current_features[0]['input_ids'] = [self.tokenizer.mask_token_id
                                                        if idx in sampled_mask_tokens
                                                        else idx for idx in current_features[0]['input_ids']]
                current_features[0]['mask_target_labels'] = current_features[0]['input_ids']

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
                return_attention_mask=True,
                return_tensors=return_tensors,
            )
        features['labels'] = torch.Tensor([feature['labels'] for feature in new_features])

        if self.coword_random_mask_rate > 0:
            features['mask_target_labels'] = self.tokenizer.pad(
                    {'input_ids': [feature['mask_target_labels'] for feature in new_features]},
                    padding=self.padding,
                    max_length=self.max_length,
                    return_tensors=return_tensors,
                )['input_ids']

        return features


class Pooler:
    """
    Using Pooler to obtain sentence embeddings.

    :param model: PreTrainedModel
    :param pooling_strategy: Optional[str].
        Currently support [`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]. Default None.
    :param padding_side: Optional[str]. `left` or `right`. Default None.
    :param is_llm: bool. Default False
    """
    def __init__(self,
                 model: PreTrainedModel,
                 pooling_strategy: Optional[Union[int, str]] = None,
                 padding_side: Optional[str] = None):
        self.model = model
        self.pooling_strategy = pooling_strategy
        self.padding_side = padding_side

    def __call__(self,
                 inputs: Dict,
                 layer_index: int = -1,
                 embedding_start: Optional[int] = None,
                 embedding_size: Optional[int] = None,
                 return_all_layer_outputs: bool = False,
                 pooling_strategy: Optional[Union[int, str]] = None,
                 return_mlm_logits: bool = False) -> torch.Tensor:
        """ Get sentence embeddings.

        :param inputs: Dict. Model inputs.
        :param layer_index: Optional[int]. Get embeddings from specific layer.
        :param embedding_start: Optional[int]. Start index of embeddings.
        :param embedding_size: int. Set embedding size for sentence embeddings.
        :param return_all_layer_outputs: bool. Return all layer outputs or not. Default False.
        :param pooling_strategy: Optional[str].
            Currently support [`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]. Default None.
        :param return_mlm_logits: bool. Return logits or not. Default False.
        """
        ret = self.model(output_hidden_states=True, return_dict=True, **inputs)
        all_layer_outputs = ret.hidden_states
        if return_all_layer_outputs:
            return (all_layer_outputs, ret.logits) if return_mlm_logits else all_layer_outputs
        outputs = all_layer_outputs[layer_index]
        outputs = get_pooling(outputs, inputs,
                              pooling_strategy or self.pooling_strategy,
                              padding_side=self.padding_side)
        n_dim = len(outputs.shape)
        if embedding_start is not None:
            if n_dim == 2:
                outputs = outputs[:, embedding_start:]
            elif n_dim == 3:
                outputs = outputs[:, :, embedding_start:]
            else:
                raise ValueError(f'Unsupported output shape: {outputs.shape}')
        if embedding_size is not None:
            # topk embedding size
            if n_dim == 2:
                outputs = outputs[:, :embedding_size]
            elif n_dim == 3:
                outputs = outputs[:, :, :embedding_size]
            else:
                raise ValueError(f'Unsupported output shape: {outputs.shape}')
        return (outputs, ret.logits) if return_mlm_logits else outputs


class AngleTrainer(Trainer):
    """
    Custom Huggingface Trainer for AnglE.

    :param pooler: Pooler. Required
    :param loss_kwargs: Optional[Dict]. Default None.
    :param dataset_format: str. Default DatasetFormats.A
    :param teacher_name_or_path: Optional[str]. For distribution alignment.
    :param **kwargs: other parameters of Trainer.
    """
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = DatasetFormats.A,
                 teacher_name_or_path: Optional[str] = None,
                 teacher_pooling_strategy: str = 'cls',
                 pad_token_id: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        self.pad_token_id = pad_token_id
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_fct = AngleLoss(dataset_format=dataset_format, **loss_kwargs)
        self.teacher_name_or_path = teacher_name_or_path
        self.teacher_pooling_strategy = teacher_pooling_strategy
        if teacher_name_or_path is not None:
            logger.info('Teacher detected! '
                        'please ensure the teacher has the same tokenizer as the backbone model!')
            assert not check_llm(teacher_name_or_path), ('Currently not support LLMs alignment,'
                                                         f' teacher={teacher_name_or_path}')
            teacher_backbone = AutoModel.from_pretrained(
                teacher_name_or_path,
                trust_remote_code=True,
                torch_dtype=self.pooler.model.dtype).to(self.pooler.model.device)

            self.teacher_pooler = Pooler(
                teacher_backbone,
                pooling_strategy=self.teacher_pooling_strategy,
                padding_side=self.pooler.padding_side)
            logger.info(f'Train with teacher={teacher_name_or_path}')

    def compute_distillation_loss(self,
                                  inputs: torch.Tensor,
                                  targets: torch.Tensor,
                                  mse_weight: float = 1.0,
                                  kl_temperature: float = 1.0) -> torch.Tensor:
        """ Compute distillation loss.

        :param inputs: torch.Tensor. Input tensor.
        :param targets: torch.Tensor. Target tensor.
        :param mse_weight: float. MSE weight. Default 1.0.
        :param kl_temperature: float. KL temperature. Default 1.0.

        :return: torch.Tensor. Distillation loss.
        """
        loss = 0.
        if mse_weight > 0:
            loss += mse_weight * nn.MSELoss()(inputs, targets)
        if kl_temperature > 0:
            loss += nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(inputs / kl_temperature, dim=-1),
                F.softmax(targets / kl_temperature, dim=-1)
            ) * kl_temperature
        return loss

    def compute_mlm_loss(self, logits, mask_target_labels):
        return F.cross_entropy(
            logits.transpose(1, 2),
            mask_target_labels,
            ignore_index=self.pad_token_id,
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        """ Compute loss for AnglE.

        :param model: Huggingface model.
        :param inputs: Dict. Model inputs.
        :param return_outputs: bool. Return outputs or not. Default False.

        :return: torch.Tensor. Loss.
        """
        labels = inputs.pop("labels", None)
        mask_target_labels = inputs.pop("mask_target_labels", None)
        if mask_target_labels is not None:
            all_layer_outputs, mlm_logits = self.pooler(
                inputs, layer_index=-1, return_all_layer_outputs=True, return_mlm_logits=True)
        else:
            all_layer_outputs = self.pooler(inputs, layer_index=-1, return_all_layer_outputs=True)
        all_outputs = all_layer_outputs[-1]
        outputs = get_pooling(all_outputs, inputs,
                              self.pooler.pooling_strategy,
                              self.pooler.padding_side)
        loss = self.loss_fct(labels, outputs)
        if self.teacher_name_or_path is not None:
            with torch.no_grad():
                self.teacher_pooler.model = self.teacher_pooler.model.to(self.pooler.model.device)
                align_outputs = self.teacher_pooler(inputs)

            alignment_loss = self.compute_distillation_loss(
                all_outputs if self.teacher_pooling_strategy == 'all' else outputs,
                align_outputs,
                mse_weight=0.0,
                kl_temperature=1.0)
            loss += alignment_loss

        if mask_target_labels is not None:
            loss += self.compute_mlm_loss(mlm_logits, mask_target_labels)

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, *args, **kwargs):
        eval_loss = self.compute_loss(model, inputs, return_outputs=False)
        return eval_loss, None, None


class AngleESETrainer(AngleTrainer):
    """
    Custom Huggingface Trainer for AnglE Espresso.

    :param pooler: Pooler. Required
    :param loss_kwargs: Optional[Dict]. Default None.
    :param dataset_format: str. Default DatasetFormats.A
    :param teacher_name_or_path: Optional[str]. For distribution alignment.
    :param **kwargs: other parameters of Trainer.
    """
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = DatasetFormats.A,
                 teacher_name_or_path: Optional[str] = None,
                 ese_kl_temperature: float = 1.0,
                 ese_compression_size: int = 128,
                 apply_ese_pca: bool = True,
                 **kwargs):
        super().__init__(pooler=pooler,
                         loss_kwargs=loss_kwargs,
                         dataset_format=dataset_format,
                         teacher_name_or_path=teacher_name_or_path,
                         **kwargs)
        self.ese_kl_temperature = ese_kl_temperature
        self.ese_compression_size = ese_compression_size
        self.apply_ese_pca = apply_ese_pca
        self.n_layers = self.pooler.model.config.num_hidden_layers
        logger.info('Train with ☕️ Espresso!')

    @torch.no_grad()
    def pca_compress(self, m: torch.Tensor, k: int) -> torch.Tensor:
        """ Get topk feature via PCA.

        :param m: torch.Tensor. Input tensor.
        :param k: int. Top-k feature size.

        :return: torch.Tensor. Top-k feature.
        """
        A = F.softmax(m.T @ m / m.shape[-1]**0.5, dim=-1)
        u, s, _ = torch.svd_lowrank(A, q=k)
        # top-k principal components
        topk_deps = u @ torch.diag(s)
        return m @ topk_deps

    def compute_student_loss(self,
                             inputs: Dict,
                             all_layer_outputs: torch.Tensor,
                             labels: torch.Tensor,
                             pooling_strategy: str,
                             padding_side: str) -> torch.Tensor:
        loss = 0.
        compression_loss = 0.
        for i in range(self.n_layers - 1):
            division = (1. + math.log(1 + i))
            all_student_outputs = all_layer_outputs[i]
            student_outputs = get_pooling(all_student_outputs,
                                          inputs,
                                          pooling_strategy,
                                          padding_side)

            slimmed_outputs = student_outputs[:, :self.ese_compression_size]
            loss += self.loss_fct(labels, slimmed_outputs) / division
            if self.apply_ese_pca:
                compression_loss += self.compute_distillation_loss(
                    slimmed_outputs,
                    self.pca_compress(student_outputs, self.ese_compression_size),
                    kl_temperature=self.ese_kl_temperature
                ) / division
        return (loss + compression_loss) / (self.n_layers - 1)

    def compute_loss(self, model, inputs, return_outputs=False):
        """ Compute loss for Espresso.

        :param model: Huggingface model.
        :param inputs: Dict. Model inputs.
        :param return_outputs: bool. Return outputs or not. Default False.

        :return: torch.Tensor. Loss.
        """
        labels = inputs.pop("labels", None)
        mask_target_labels = inputs.pop("mask_target_labels", None)
        # layer
        if mask_target_labels is not None:
            all_layer_outputs, mlm_logits = self.pooler(
                inputs, layer_index=-1, return_all_layer_outputs=True, return_mlm_logits=True)
        else:
            all_layer_outputs = self.pooler(inputs, layer_index=-1, return_all_layer_outputs=True)
        all_teacher_outputs = all_layer_outputs[-1]
        teacher_outputs = get_pooling(all_teacher_outputs, inputs,
                                      self.pooler.pooling_strategy,
                                      self.pooler.padding_side)

        loss = self.loss_fct(labels, teacher_outputs)

        slimmed_outputs = teacher_outputs[:, :self.ese_compression_size]
        loss += self.loss_fct(labels, slimmed_outputs)
        if self.apply_ese_pca:
            loss += self.compute_distillation_loss(
                slimmed_outputs,
                self.pca_compress(teacher_outputs, self.ese_compression_size),
                kl_temperature=self.ese_kl_temperature
            )

        # student loss
        loss += self.compute_student_loss(
            inputs,
            all_layer_outputs,
            labels,
            self.pooler.pooling_strategy,
            self.pooler.padding_side,
        )

        # alignment loss
        if self.teacher_name_or_path is not None:
            with torch.no_grad():
                self.teacher_pooler.model = self.teacher_pooler.model.to(self.pooler.model.device)
                align_outputs = self.teacher_pooler(inputs)
                alignment_loss = self.compute_distillation_loss(
                    all_teacher_outputs if self.teacher_pooling_strategy == 'all' else teacher_outputs,
                    align_outputs,
                    mse_weight=0.0,
                    kl_temperature=1.0
                )
                loss += alignment_loss

        if mask_target_labels is not None:
            loss += self.compute_mlm_loss(mlm_logits, mask_target_labels)

        return (loss, teacher_outputs) if return_outputs else loss


class AngleLoss:
    """
    Configure AngleLoss.

    :param cosine_w: float. weight for cosine_loss. Default 1.0
    :param ibn_w: float. weight for contrastive loss. Default 1.0
    :param angle_w: float. weight for angle loss. Default 1.0
    :param cosine_tau: float. tau for cosine loss. Default 20.0
    :param ibn_tau: float. tau for contrastive loss. Default 20.0
    :param angle_tau: float. tau for angle loss. Default 20.0
    :param angle_pooling_strategy: str. pooling strategy for angle loss. Default'sum'.
    :param dataset_format: Optional[str]. Default None.
    """
    def __init__(self,
                 cosine_w: float = 0.0,
                 ibn_w: float = 20.0,
                 angle_w: float = 1.0,
                 cosine_tau: float = 20.0,
                 ibn_tau: float = 20.0,
                 angle_tau: float = 20.0,
                 angle_pooling_strategy: str = 'sum',
                 dataset_format: Optional[str] = None,
                 **kwargs):
        if 'w1' in kwargs or 'w2' in kwargs or 'w3' in kwargs:
            assert ('w1, w2, and w3 has been renamed to cosine_w, ibn_w, and angle_w, respecitvely.'
                    'Please use new names instead.')
        self.cosine_w = cosine_w
        self.ibn_w = ibn_w
        self.angle_w = angle_w
        self.cosine_tau = cosine_tau
        self.ibn_tau = ibn_tau
        self.angle_tau = angle_tau
        self.angle_pooling_strategy = angle_pooling_strategy
        self.dataset_format = dataset_format

    def __call__(self,
                 labels: torch.Tensor,
                 outputs: torch.Tensor) -> torch.Tensor:
        """ Compute loss for AnglE.

        :param labels: torch.Tensor. Labels.
        :param outputs: torch.Tensor. Outputs.

        :return: torch.Tensor. Loss.
        """
        if self.dataset_format == DatasetFormats.A:
            loss = 0.
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(labels, outputs, self.cosine_tau)
            if self.ibn_w > 0:
                loss += self.ibn_w * in_batch_negative_loss(labels, outputs, self.ibn_tau)
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(labels, outputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
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
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(combined_labels, combined_inputs, self.cosine_tau)
            if self.ibn_w > 0:
                loss += self.ibn_w * contrastive_with_negative_loss(text, positive, negative, tau=self.ibn_tau)
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(combined_labels, combined_inputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
        elif self.dataset_format == DatasetFormats.C:
            text = outputs[::2]
            positive = outputs[1::2]
            loss = contrastive_with_negative_loss(text, positive, neg=None, tau=self.ibn_tau)
        else:
            raise NotImplementedError
        return loss


class AnglE(AngleBase):
    """
    AnglE. Everything is here👋

    :param model_name_or_path: str, model name or path.
    :param tokenizer_name_or_path: Optional[str]. Default None. When it set to None, it will use the same as `model_name_or_path`.
    :param max_length: int. Default 512
    :param model_kwargs: Optional[Dict]. kwargs for model.
    :param lora_config_kwargs: Optional[Dict]. kwargs for peft lora_config.
        details refer to: https://huggingface.co/docs/peft/tutorial/peft_model_config
    :param pooling_strategy: Optional[str]. Pooling strategy.
        Currently support [`cls`, `cls_avg`, `cls_max`, `last`, `avg`, `mean`, `max`, `all`, int]
    :param apply_lora: Optional[bool]. Whether apply lora. Default None.
    :param train_mode: bool. Whether load for training. Default True.
    :param load_kbit: Optional[int]. Specify kbit training from [4, 8, 16]. Default None.
    :param is_llm: Optional[bool]. Whether the model is llm. Default None.
    :param pretrained_model_path: Optional[str]. Default None.
    :param pretrained_lora_path: Optional[str]. Default None.
    :param torch_dtype: Optional[torch.dtype]. Specify torch_dtype. Default None.
    :param device: Optional[str]. Specify device. Default None.
    :param kbit_kwargs: Optional[Dict]. kwargs for kbit. Default None.
        details refer to: https://huggingface.co/docs/peft/package_reference/peft_model#peft.prepare_model_for_kbit_training
    :param tokenizer_padding_side: Optional[str]. Specify tokenizer padding side from [`left`, `right`]. Default None.
    :param apply_billm: bool. Whether apply billm. Default False.
    :param billm_model_class: Optional[str]. Specify billm model class. Default None.
    :param load_mlm_model: bool. Whether load mlm model. Default False. If set True, it will load model with AutoModelForMaskedLM.
    :param **kwargs: Any.
    """  # NOQA
    cfg_file_name = 'angle_config.json'
    special_columns = ['labels', 'seperate_ids', 'extra', 'mask_target_labels']

    def __init__(self,
                 model_name_or_path: str,
                 tokenizer_name_or_path: Optional[str] = None,
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
                 torch_dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None,
                 kbit_kwargs: Optional[Dict] = None,
                 tokenizer_padding_side: Optional[str] = None,
                 apply_billm: bool = False,
                 billm_model_class: Optional[str] = None,
                 load_mlm_model: bool = False,
                 **kwargs: Any):
        super().__init__()
        self.max_length = max_length
        self.train_mode = train_mode
        self.pooling_strategy = pooling_strategy
        self.load_kbit = load_kbit
        self.is_llm = is_llm
        self.load_mlm_model = load_mlm_model
        if device:
            self.device = device
        else:
            self.device = set_device()
        if is_llm is None:
            self.is_llm = check_llm(model_name_or_path)
            if self.is_llm:
                logger.info('LLM detected, automatically set is_llm=True.'
                            'If it is wrong, you can manually set `is_llm`.')
        if self.is_llm and self.pooling_strategy != 'last':
            logger.info(f'🚨 LLM detected, but pooling strategy is specified to {self.pooling_strategy}.'
                        'Please check whether it is correct. It is recommended to use `last` pooling strategy for LLM.')

        self.apply_lora = apply_lora
        if self.apply_lora is None:
            if self.is_llm:
                self.apply_lora = True
                logger.info('LLM detected, automatically set apply_lora=True.'
                            'If it is wrong, you can manually set `apply_lora`.')
            if pretrained_lora_path is not None:
                self.apply_lora = True

        if self.device == 'cuda':
            self.gpu_count = torch.cuda.device_count()
        elif self.device == 'mps':
            self.gpu_count = 1
        else:
            self.gpu_count = 0

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

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path, trust_remote_code=True)
        if tokenizer_padding_side is not None and self.tokenizer.padding_side != tokenizer_padding_side:
            self.tokenizer.padding_side = tokenizer_padding_side
        if self.is_llm and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        kbit_kwargs = kbit_kwargs if kbit_kwargs is not None else {}
        if self.is_llm:
            device_map = "auto"
            if apply_billm:
                assert billm_model_class is not None, "billm_model_class should be specified for apply_billm=True"
                try:
                    import billm
                except ImportError as err:
                    print(f'Import Error: {err}')
                    print('Please install the latest billm via: python -m pip install -U billm')
                    raise

                MODEL_CLASS = getattr(billm, billm_model_class)
            else:
                MODEL_CLASS = AutoModelForCausalLM
            if train_mode and self.gpu_count > 1:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            # LLM
            if self.apply_lora:
                lora_config['bias'] = "none"
                lora_config['task_type'] = TaskType.CAUSAL_LM

                is_kbit = load_kbit in [4, 8]
                if is_kbit:
                    model = MODEL_CLASS.from_pretrained(
                        model_name_or_path,
                        config=None,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=load_kbit == 4,
                            load_in_8bit=load_kbit == 8,
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
                else:
                    model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                        device_map=device_map,
                                                        output_hidden_states=True,
                                                        trust_remote_code=True,
                                                        torch_dtype=torch_dtype or torch.float16)
                if train_mode and is_kbit:
                    model = prepare_model_for_kbit_training(model, **kbit_kwargs)

                if pretrained_lora_path is not None:
                    logger.info(f'Load lora weight from {pretrained_lora_path}')
                    model = PeftModel.from_pretrained(
                        model,
                        pretrained_lora_path,
                        torch_dtype=torch.float32 if is_kbit else (torch_dtype or torch.float16),
                        device_map=device_map,
                        is_trainable=train_mode
                    )
                elif train_mode:
                    if 'target_modules' not in lora_config or lora_config.get('target_modules', None) is None:
                        target_modules = find_all_linear_names(
                            model, linear_type=bnb.nn.Linear4bit if load_kbit == 4 else nn.Linear)
                        lora_config['target_modules'] = target_modules
                        logger.info(f'lora target modules={target_modules}')
                    peft_config = LoraConfig(**lora_config)
                    model = get_peft_model(model, peft_config)

                if is_kbit:
                    model = AnglE.kbit_post_handle(model)

                self.backbone = model
            else:
                model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                    device_map=device_map,
                                                    output_hidden_states=True,
                                                    trust_remote_code=True,
                                                    torch_dtype=torch_dtype or torch.float16)
                self.backbone = model
        else:
            MODEL_CLASS = AutoModelForMaskedLM if load_mlm_model else AutoModel
            # non-LLMs
            if self.apply_lora:
                model = MODEL_CLASS.from_pretrained(pretrained_model_path or model_name_or_path, trust_remote_code=True)
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
                self.backbone = MODEL_CLASS.from_pretrained(
                    pretrained_model_path or model_name_or_path,
                    trust_remote_code=True)

        if train_mode and self.apply_lora:
            self.backbone.print_trainable_parameters()

        self.backbone.config.use_cache = False
        self.pooler = Pooler(
            self.backbone,
            pooling_strategy=self.pooling_strategy,
            padding_side=self.tokenizer.padding_side)

        self.__cfg = {
            'model_name_or_path': model_name_or_path,
            'max_length': max_length,
            'model_kwargs': model_kwargs,
            'pooling_strategy': pooling_strategy,
            'lora_config_kwargs': lora_config,
            'is_llm': self.is_llm,
            'apply_billm': apply_billm,
            'billm_model_class': billm_model_class,
            'apply_lora': self.apply_lora,
            'tokenizer_padding_side': tokenizer_padding_side,
            'angle_emb_version': __version__,
        }
        self.__cfg.update(kwargs)

    def cuda(self):
        if self.gpu_count > 1 and self.is_llm:
            return self
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
            valid_ds_for_callback: Optional[Dataset] = None,
            batch_size: int = 32,
            output_dir: Optional[str] = None,
            epochs: int = 1,
            learning_rate: float = 1e-5,
            warmup_steps: int = 1000,
            logging_steps: int = 10,
            eval_steps: int = 1000,
            evaluation_strategy: str = 'steps',
            save_steps: int = 100,
            save_strategy: str = 'steps',
            save_total_limit: int = 1,
            gradient_accumulation_steps: int = 1,
            fp16: Optional[bool] = None,
            argument_kwargs: Optional[Dict] = None,
            trainer_kwargs: Optional[Dict] = None,
            loss_kwargs: Optional[Dict] = None,
            apply_ese: bool = False,
            filter_duplicate: bool = True,
            push_to_hub: bool = False,
            hub_model_id: Optional[str] = None,
            hub_private_repo: bool = True,
            coword_random_mask_rate: float = 0.,
            padding: str = 'longest'):
        """
        Fit using AnglE.

        :param train_ds: Dataset. tokenized train dataset. Required.
        :param valid_ds: Optional[Dataset]. tokenized valid dataset. Default None.
        :param valid_ds_for_callback: Optional[Dataset]. tokenized valid dataset for callback use.
            The dataset format should be `DatasetFormats.A`. The spearmans' correlation will be computed
            after each epoch training and the best model will be saved. Default None.
        :param batch_size: int. Default 32.
        :param output_dir: Optional[str]. save dir. Default None.
        :param epochs: int. Default 1.
        :param learning_rate: float. Default 1e-5.
        :param warmup_steps: int. Default 1000.
        :param logging_steps: int. Default 10.
        :param eval_steps: int. Default 1000.
        :param evaluation_strategy: str. Default 'steps'.
        :param save_steps: int. Default 100.
        :param save_strategy: str. Default steps.
        :param save_total_limit: int. Default 10.
        :param gradient_accumulation_steps: int. Default 1.
        :param fp16: Optional[bool]. Default None.
        :param argument_kwargs: Optional[Dict]. kwargs for TrainingArguments.
            refer to: https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments
        :param trainer_kwargs: Optional[Dict]. kwargs for AngleTrainer.
        :param loss_kwargs: Optional[Dict]. kwargs for AngleLoss.
        :param apply_ese: bool, whether apply ESE training.
        :param filter_duplicate: bool, whether filter duplicate samples.
        :param push_to_hub: bool, whether push to hub.
        :param hub_model_id: Optional[str], hub model id.
        :param hub_private_repo: bool, whether push to private repo.
        :param coword_random_mask_rate: float, random mask common token rate. Default 0.
        :param padding: str, padding strategy of tokenizer. Default 'longest'.
        """  # NOQA
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        # save config
        self.save_config(os.path.join(output_dir, AnglE.cfg_file_name))
        # save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        if self.gpu_count > 1:
            gradient_accumulation_steps = gradient_accumulation_steps // self.gpu_count
        if fp16 is None and self.is_llm:
            fp16 = True
        else:
            fp16 = False

        # init argument_kwargs
        if argument_kwargs is None:
            argument_kwargs = {}
        if 'push_to_hub' not in argument_kwargs:
            argument_kwargs['push_to_hub'] = push_to_hub
        if 'hub_model_id' not in argument_kwargs:
            argument_kwargs['hub_model_id'] = hub_model_id
        if 'hub_private_repo' not in argument_kwargs:
            argument_kwargs['hub_private_repo'] = hub_private_repo

        if trainer_kwargs is None:
            trainer_kwargs = {}

        callbacks = None
        if valid_ds_for_callback is not None:
            # check format
            for obj in valid_ds_for_callback:
                if obj['extra']['dataset_format'] != DatasetFormats.A:
                    raise ValueError('Currently only support evaluation for DatasetFormats.A.')
                break
            best_ckpt_dir = None
            if output_dir is not None:
                best_ckpt_dir = os.path.join(output_dir, 'best-checkpoint')
            evaluate_callback = EvaluateCallback(self, valid_ds_for_callback,
                                                 partial(self.evaluate, batch_size=batch_size),
                                                 save_dir=best_ckpt_dir,
                                                 push_to_hub=push_to_hub,
                                                 hub_model_id=hub_model_id,
                                                 hub_private_repo=hub_private_repo)
            # set False to ensure only best checkpoint is pushed
            argument_kwargs['push_to_hub'] = False
            callbacks = [evaluate_callback]

        if coword_random_mask_rate > 0:
            logger.info(f'Trained with random mask common tokens.'
                        f'coword_random_mask_rate={coword_random_mask_rate}')
        CustomTrainer = AngleESETrainer if apply_ese else AngleTrainer
        trainer = CustomTrainer(
            pooler=self.pooler,
            model=self.backbone,
            dataset_format=self.detect_dataset_format(train_ds),
            train_dataset=train_ds,
            eval_dataset=valid_ds,
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
                save_steps=save_steps,
                save_strategy=save_strategy,
                evaluation_strategy=evaluation_strategy if valid_ds is not None else 'no',
                eval_steps=eval_steps,
                output_dir=output_dir,
                save_total_limit=save_total_limit,
                load_best_model_at_end=False,
                ddp_find_unused_parameters=False if self.gpu_count > 1 else None,
                remove_unused_columns=False,
                **argument_kwargs,
            ),
            callbacks=callbacks,
            data_collator=AngleDataCollator(
                self.tokenizer,
                padding=padding,
                return_tensors="pt",
                max_length=self.max_length,
                filter_duplicate=filter_duplicate,
                coword_random_mask_rate=coword_random_mask_rate,
            ),
            **trainer_kwargs
        )
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.backbone = torch.compile(self.backbone)

        trainer.train()
        if argument_kwargs.get('push_to_hub', False):
            trainer.push_to_hub()
        self.backbone.save_pretrained(output_dir)

    def evaluate(self, data: Dataset, batch_size: int = 32, metric: str = 'spearman_cosine') -> float:
        """ evaluate

        :param data: Dataset, DatasetFormats.A is required
        :param batch_size: int. Default 32.
        :param metric: str. Default 'spearman_cosine'.

        :return: float.
        """
        return CorrelationEvaluator(
            text1=data['text1'],
            text2=data['text2'],
            labels=data['label'],
            batch_size=batch_size,
        )(self)[metric]

    def truncate_layer(self, layer_index: int):
        """ truncate layer

        :param layer_index: int. layers after layer_index will be truncated.
        :return: self
        """
        if len(self.backbone.encoder.layer) < layer_index:
            logger.info('current layer_index is larger than the number of layers, please check whether it is correct')
        self.backbone.encoder.layer = self.backbone.encoder.layer[:layer_index]
        return self

    def encode(self,
               inputs: Union[List[str], Tuple[str], List[Dict], str],
               max_length: Optional[int] = None,
               end_with_eos: bool = False,
               to_numpy: bool = True,
               embedding_start: int = 0,
               embedding_size: Optional[int] = None,
               device: Optional[Any] = None,
               prompt: Optional[str] = None,
               normalize_embedding: bool = False,
               padding: str = 'longest'):
        """
        encode texts.

        :param inputs: Union[List[str], Tuple[str], List[Dict], str]. Input texts. Required.
        :param max_length: Optional[int]. Default None.
        :param to_numpy: bool. Default True.
        :param embedding_start: int. Specify the start position of the embedding (for Espresso).
        :param embedding_size: Optional[int]. Specify embedding size (for Espresso).
            The embeddings from embedding_start to embedding_start+embedding_size will be returned.
        :param device: Optional[Any]. Default None.
        :param prompt: Optional[str]. Default None.
        :param normalize_embedding: bool. Default False.
        :param padding: str. Padding strategy of tokenizer. Default 'longest'.
        """
        self.backbone.eval()

        if device is None:
            device = self.device
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if prompt is not None:
            for i, obj in enumerate(inputs):
                assert isinstance(obj, dict), 'The prompt has been set, please pass a dict like {"prompt_key": "text"}'
                inputs[i] = prompt.format(**obj)
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
            tok = self.tokenizer.pad(tok, padding=padding, return_attention_mask=True, return_tensors='pt')
        else:
            tok = self.tokenizer(
                inputs,
                padding=padding,
                max_length=max_length or self.max_length,
                truncation=True,
                return_tensors='pt')
        tok.to(device)
        with torch.no_grad():
            output = self.pooler(tok,
                                 embedding_start=embedding_start,
                                 embedding_size=embedding_size)
        if normalize_embedding:
            output = nn.functional.normalize(output, p=2, dim=-1)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output

    def push_to_hub(self, hub_model_id: str, private: bool = True, exist_ok: bool = False, **kwargs):
        """ push model to hub

        :param hub_model_id: str, hub model id.
        :param private: bool, whether push to private repo. Default True.
        :param exist_ok: bool, whether allow overwrite. Default False.
        :param kwargs: other kwargs for `push_to_hub` method.
        """
        if not exist_ok and repo_exists(hub_model_id):
            raise ValueError(f"Model {hub_model_id} already exists on the hub. Set `exist_ok=True` to overwrite.")
        self.tokenizer.push_to_hub(hub_model_id, private=private, **kwargs)
        self.backbone.push_to_hub(hub_model_id, private=private, **kwargs)

    def save_pretrained(self, output_dir: str, exist_ok: bool = True):
        """ save model and tokenizer

        :param output_dir: str, output dir.
        :param exist_ok: bool, whether allow overwrite. Default True.
        """
        if not exist_ok and os.path.exists(output_dir):
            raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")
        os.makedirs(output_dir, exist_ok=exist_ok)
        self.tokenizer.save_pretrained(output_dir)
        self.backbone.save_pretrained(output_dir)


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
                 model: AnglE,
                 valid_ds: Dataset,
                 evaluate_fn: Callable,
                 save_dir: Optional[str] = None,
                 push_to_hub: bool = False,
                 hub_model_id: Optional[str] = None,
                 hub_private_repo: bool = True):
        super().__init__()
        self.model = model
        self.valid_ds = valid_ds
        self.evaluate_fn = evaluate_fn
        self.save_dir = save_dir
        self.best_corrcoef = 0
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id
        self.hub_private_repo = hub_private_repo

    def on_epoch_end(self, args, state, control, **kwargs):
        corrcoef = self.evaluate_fn(self.valid_ds)
        if corrcoef > self.best_corrcoef:
            self.best_corrcoef = corrcoef
            print('new best corrcoef!')
            if self.save_dir is not None:
                self.model.save_pretrained(self.save_dir)
                print(f'save to {self.save_dir}')
                if self.push_to_hub:
                    self.model.push_to_hub(
                        self.hub_model_id,
                        private=self.hub_private_repo,
                        exist_ok=True,
                        commit_message='new best checkpoint')
        logger.info(f'corrcoef: {corrcoef}, best corrcoef: {self.best_corrcoef}')
