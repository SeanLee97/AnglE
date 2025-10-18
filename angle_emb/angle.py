import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import repo_exists
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from .base import AngleBase
from .evaluation import CorrelationEvaluator
from .loss import (
    angle_loss,
    contrastive_with_negative_loss,
    cosine_loss,
    in_batch_negative_loss,
)
from .utils import find_all_linear_names, get_pooling, logger, set_device
from .version import __version__


def detect_dataset_format(ds: Dataset) -> str:
    """Detect dataset format from raw data"""
    columns = ds.column_names
    if 'text1' in columns and 'text2' in columns and 'label' in columns:
        return 'A'
    elif 'query' in columns and 'positive' in columns and 'negative' in columns:
        return 'C'
    elif 'query' in columns and 'positive' in columns:
        return 'B'
    else:
        raise NotImplementedError('Unable to detect dataset format')


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


@dataclass
class AngleDataCollator:
    """
    AngleDataCollator. It will be implicitly used in AnglE.fit().
    It handles raw data, tokenizes it, and prepares batches.

    :param tokenizer:  PreTrainedTokenizerBase
    :param padding:   Union[bool, str, PaddingStrategy], padding strategy
    :param max_length:  Optional[int], max length
    :param return_tensors:  str
    :param filter_duplicate: bool. Whether filter duplicate data
    :param text_prompt: Optional[str], prompt for text1 and text2 (format A only). Default None
    :param query_prompt: Optional[str], prompt for query field. Default None
    :param doc_prompt: Optional[str], prompt for positive/negative fields. Default None
    :param dataset_format: Optional[str]. Specify dataset_format: 'A', 'B', or 'C'. Default None.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'longest'
    max_length: Optional[int] = None
    return_tensors: str = "pt"
    filter_duplicate: bool = True
    text_prompt: Optional[str] = None
    query_prompt: Optional[str] = None
    doc_prompt: Optional[str] = None
    dataset_format: Optional[str] = None

    @staticmethod
    def sample_from_list(text: Union[str, List[str]]) -> str:
        """Sample one string from list or return string as is"""
        if isinstance(text, list):
            return random.choice(text)
        return text

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """ Collate function that handles raw data.

        :param features: List[Dict]. Raw data samples
        :param return_tensors: str. Default "pt"
        :return: Dict[str, torch.Tensor]. Collated data
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Auto-detect dataset format from first sample
        if self.dataset_format is None:
            sample = features[0]
            if 'text1' in sample and 'text2' in sample and 'label' in sample:
                self.dataset_format = 'A'
                logger.info('Detect dataset format A: text1, text2, label')
            elif 'query' in sample and 'positive' in sample and 'negative' in sample:
                self.dataset_format = 'C'
                logger.info('Detect dataset format C: query, positive, negative')
            elif 'query' in sample and 'positive' in sample and 'negative' not in sample:
                self.dataset_format = 'B'
                logger.info('Detect dataset format B: query, positive')
            else:
                raise NotImplementedError(
                    'Currently only support three dataset formats: '
                    'Format A: must include three columns: `text1`, `text2`, and `label`. '
                    'Format B: must include two columns: `query`, `positive`. '
                    'Format C: must include three columns: `query`, `positive`, `negative`.'
                )
        
        # Process features based on format
        processed_features = []
        duplicate_set = set()

        for feature in features:
            texts = []
            label = -1

            if self.dataset_format == 'A':
                # Format A: text1, text2, label
                text1 = self.sample_from_list(feature['text1'])
                text2 = self.sample_from_list(feature['text2'])
                label = float(feature['label'])

                # Apply text_prompt if provided (only for format A)
                if self.text_prompt is not None:
                    text1 = self.text_prompt.format(text=text1)
                    text2 = self.text_prompt.format(text=text2)

                texts = [text1, text2]

            elif self.dataset_format == 'B':
                # Format B: query, positive
                query = self.sample_from_list(feature['query'])
                positive = self.sample_from_list(feature['positive'])

                # Apply prompts
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)

                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)

                texts = [query, positive]

            elif self.dataset_format == 'C':
                # Format C: query, positive, negative
                query = self.sample_from_list(feature['query'])
                positive = self.sample_from_list(feature['positive'])
                negative = self.sample_from_list(feature['negative'])

                # Apply prompts
                if self.query_prompt is not None:
                    query = self.query_prompt.format(text=query)

                if self.doc_prompt is not None:
                    positive = self.doc_prompt.format(text=positive)
                    negative = self.doc_prompt.format(text=negative)

                texts = [query, positive, negative]

            # Tokenize texts
            tokenized_texts = []
            is_duplicate = False
            for text in texts:
                tok = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=True
                )

                # Check for duplicates
                input_ids_tuple = tuple(tok['input_ids'])
                if self.filter_duplicate and input_ids_tuple in duplicate_set:
                    is_duplicate = True
                    break
                duplicate_set.add(input_ids_tuple)

                tok['labels'] = [label]
                tokenized_texts.append(tok)

            if self.filter_duplicate and is_duplicate:
                continue

            processed_features.extend(tokenized_texts)

        # Pad and convert to tensors
        if not processed_features:
            raise ValueError('No features to process, please considering disabling filter_duplicate')

        batch = self.tokenizer.pad(
            {'input_ids': [f['input_ids'] for f in processed_features]},
            padding=self.padding,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        batch['labels'] = torch.Tensor([f['labels'] for f in processed_features])

        return batch


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
        if layer_index == -1 and not return_all_layer_outputs:
            ret = self.model(output_hidden_states=True, **inputs)
            outputs = ret.last_hidden_state if hasattr(ret, 'last_hidden_state') else ret.hidden_states[-1]
        else:
            ret = self.model(output_hidden_states=True, return_dict=True, **inputs)
            all_layer_outputs = list(ret.hidden_states)
            if hasattr(ret, 'last_hidden_state'):
                all_layer_outputs[-1] = ret.last_hidden_state
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
    :param dataset_format: str. Default 'A'
    :param teacher_name_or_path: Optional[str]. For distribution alignment.
    :param **kwargs: other parameters of Trainer.
    """
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = 'A',
                 teacher_name_or_path: Optional[str] = None,
                 teacher_pooling_strategy: str = 'cls',
                 pad_token_id: int = 0,
                 model_kwargs: Optional[Dict] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.pooler = pooler
        self.pad_token_id = pad_token_id
        self.model_kwargs = model_kwargs
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_fct = AngleLoss(dataset_format=dataset_format, **loss_kwargs)
        self.teacher_name_or_path = teacher_name_or_path
        self.teacher_pooling_strategy = teacher_pooling_strategy
        if teacher_name_or_path is not None:
            logger.info('Teacher detected! '
                        'please ensure the teacher has the same tokenizer as the backbone model!')
            teacher_backbone = AutoModel.from_pretrained(
                teacher_name_or_path,
                trust_remote_code=True,
                torch_dtype=self.pooler.model.dtype,
                **self.model_kwargs).to(self.pooler.model.device)

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

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
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
    :param dataset_format: str. Default 'A'
    :param teacher_name_or_path: Optional[str]. For distribution alignment.
    :param **kwargs: other parameters of Trainer.
    """
    def __init__(self,
                 pooler: Pooler,
                 loss_kwargs: Optional[Dict] = None,
                 dataset_format: str = 'A',
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
    :param ibn_w: float. weight for in batch negative loss. Default 1.0
    :param cln_w: float. weight for contrastive learning with hard negative. Default 1.0
    :param angle_w: float. weight for angle loss. Default 1.0
    :param cosine_tau: float. tau for cosine loss. Default 20.0
    :param ibn_tau: float. tau for in batch negative loss. Default 20.0
    :param angle_tau: float. tau for angle loss. Default 20.0
    :param angle_pooling_strategy: str. pooling strategy for angle loss. Default'sum'.
    :param dataset_format: Optional[str]. Default None.
    """
    def __init__(self,
                 cosine_w: float = 0.0,
                 ibn_w: float = 1.0,
                 cln_w: float = 1.0,
                 angle_w: float = 0.02,
                 cosine_tau: float = 20.0,
                 ibn_tau: float = 20.0,
                 angle_tau: float = 20.0,
                 angle_pooling_strategy: str = 'sum',
                 dataset_format: Optional[str] = None,
                 **kwargs):
        self.cosine_w = cosine_w
        self.ibn_w = ibn_w
        self.cln_w = cln_w
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
        if self.dataset_format == 'A':
            loss = 0.
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(labels, outputs, self.cosine_tau)
            if self.ibn_w > 0:
                loss += self.ibn_w * in_batch_negative_loss(labels, outputs, self.ibn_tau)
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(labels, outputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
        elif self.dataset_format == 'B':
            # Format B: query, positive (no negative)
            query = outputs[::2]
            positive = outputs[1::2]
            loss = contrastive_with_negative_loss(query, positive, neg=None, tau=self.ibn_tau)
            
        elif self.dataset_format == 'C':
            # Format C: query, positive, negative
            if int(self.cln_w) == 0:
                logger.info('`cln_w` is set to zero. Contrastive learning with hard negative is disabled. '
                            'Please manually check whether it is correct.')
            
            query = outputs[::3]
            positive = outputs[1::3]
            negative = outputs[2::3]
            assert query.shape == positive.shape == negative.shape, f'query.shape={query.shape}, positive.shape={positive.shape}, negative.shape={negative.shape}'  # NOQA

            _, fea_dim = query.shape
            positive_inputs = torch.stack((query, positive), dim=1).reshape(-1, fea_dim)  # zip(query, positive)
            positive_labels = torch.ones_like(positive_inputs[:, :1]).long()
            negative_inputs = torch.stack((query, negative), dim=1).reshape(-1, fea_dim)  # zip(query, negative)
            negative_labels = torch.zeros_like(negative_inputs[:, :1]).long()
            combined_inputs = torch.cat((positive_inputs, negative_inputs), dim=0)
            combined_labels = torch.cat((positive_labels, negative_labels), dim=0)

            loss = 0.
            # contrastive learning loss
            cll = 0.
            if self.ibn_w > 0:
                cll += self.ibn_w * contrastive_with_negative_loss(query, positive, tau=self.ibn_tau)
            if self.cln_w > 0:
                cll += self.cln_w * contrastive_with_negative_loss(query, positive, negative, tau=self.ibn_tau)
            if cll > 0:
                loss += cll / 2
            # angle loss
            if self.angle_w > 0:
                loss += self.angle_w * angle_loss(combined_labels, combined_inputs, self.angle_tau,
                                                  pooling_strategy=self.angle_pooling_strategy)
            # cosine loss
            if self.cosine_w > 0:
                loss += self.cosine_w * cosine_loss(combined_labels, combined_inputs, self.cosine_tau)
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
    :param is_llm: Optional[bool]. Whether the model is llm. Must be set manually. Default None.
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
    special_columns = ['labels']

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
        if device is not None:
            self.device = device
        else:
            self.device = set_device()

        if self.is_llm and self.pooling_strategy != 'last':
            logger.info(f'🚨 LLM mode is enabled, but pooling strategy is specified to {self.pooling_strategy}.'
                        'Please check whether it is a correct behavior.')

        self.apply_lora = apply_lora
        if self.apply_lora is None:
            if pretrained_lora_path is not None:
                logger.info('pretrained_lora_path is specified, automatically set apply_lora=True.')
                self.apply_lora = True

        if self.device == 'cuda':
            self.gpu_count = torch.cuda.device_count()
        elif self.device == 'mps':
            self.gpu_count = 1
        else:
            self.gpu_count = 0

        if torch_dtype is None:
            torch_dtype = torch.float32 if train_mode else None

        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

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
                        **self.model_kwargs
                    )
                else:
                    model = MODEL_CLASS.from_pretrained(model_name_or_path,
                                                        device_map=device_map,
                                                        output_hidden_states=True,
                                                        trust_remote_code=True,
                                                        torch_dtype=torch_dtype or torch.float16,
                                                        **self.model_kwargs)
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
                                                    torch_dtype=torch_dtype or torch.float16,
                                                    **self.model_kwargs)
                self.backbone = model
        else:
            MODEL_CLASS = AutoModelForMaskedLM if load_mlm_model else AutoModel
            # non-LLMs
            if self.apply_lora:
                model = MODEL_CLASS.from_pretrained(
                    pretrained_model_path or model_name_or_path,
                    trust_remote_code=True,
                    **self.model_kwargs)
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
                    trust_remote_code=True,
                    **self.model_kwargs)

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
                        model_kwargs: Optional[Dict] = None,
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
        kwargs['model_kwargs'] = model_kwargs
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
            eval_strategy: str = 'steps',
            save_steps: int = 100,
            save_strategy: str = 'steps',
            save_total_limit: int = 1,
            gradient_accumulation_steps: int = 1,
            fp16: Optional[bool] = None,
            bf16: Optional[bool] = None,
            argument_kwargs: Optional[Dict] = None,
            trainer_kwargs: Optional[Dict] = None,
            loss_kwargs: Optional[Dict] = None,
            apply_ese: bool = False,
            filter_duplicate: bool = True,
            push_to_hub: bool = False,
            hub_model_id: Optional[str] = None,
            hub_private_repo: bool = True,
            padding: str = 'longest',
            text_prompt: Optional[str] = None,
            query_prompt: Optional[str] = None,
            doc_prompt: Optional[str] = None):
        """
        Fit using AnglE.

        :param train_ds: Dataset. Raw train dataset (not tokenized). Required.
        :param valid_ds: Optional[Dataset]. Raw valid dataset (not tokenized). Default None.
        :param valid_ds_for_callback: Optional[Dataset]. Raw valid dataset for callback use (not tokenized).
            The dataset format should be format A. The spearmans' correlation will be computed
            after each epoch training and the best model will be saved. Default None.
        :param batch_size: int. Default 32.
        :param output_dir: Optional[str]. save dir. Default None.
        :param epochs: int. Default 1.
        :param learning_rate: float. Default 1e-5.
        :param warmup_steps: int. Default 1000.
        :param logging_steps: int. Default 10.
        :param eval_steps: int. Default 1000.
        :param eval_strategy: str. Default 'steps'.
        :param save_steps: int. Default 100.
        :param save_strategy: str. Default steps.
        :param save_total_limit: int. Default 10.
        :param gradient_accumulation_steps: int. Default 1.
        :param fp16: Optional[bool]. Default None.
        :param bf16: Optional[bool]. Default None.
        :param argument_kwargs: Optional[Dict]. kwargs for TrainingArguments.
            refer to: https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments
        :param trainer_kwargs: Optional[Dict]. kwargs for AngleTrainer.
        :param loss_kwargs: Optional[Dict]. kwargs for AngleLoss.
        :param apply_ese: bool, whether apply ESE training.
        :param filter_duplicate: bool, whether filter duplicate samples.
        :param push_to_hub: bool, whether push to hub.
        :param hub_model_id: Optional[str], hub model id.
        :param hub_private_repo: bool, whether push to private repo.
        :param padding: str, padding strategy of tokenizer. Default 'longest'.
        :param text_prompt: Optional[str], prompt for text1 and text2 (format A only). Default None.
        :param query_prompt: Optional[str], prompt template for query. Default None.
        :param doc_prompt: Optional[str], prompt template for documents (positive/negative). Default None.
        """  # NOQA
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        # save config
        self.save_config(os.path.join(output_dir, AnglE.cfg_file_name))
        # save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        if fp16 is None:
            fp16 = False
        if bf16 is None:
            bf16 = False

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
            # check format - must be format A
            detected_format = detect_dataset_format(valid_ds_for_callback)
            if detected_format != 'A':
                raise ValueError('Currently only support evaluation for format A (text1, text2, label).')
            best_ckpt_dir = None
            if output_dir is not None:
                best_ckpt_dir = os.path.join(output_dir, 'best-checkpoint')
            evaluate_callback = EvaluateCallback(self,
                                                 valid_ds_for_callback,
                                                 partial(self.evaluate, batch_size=batch_size, prompt=text_prompt),
                                                 save_dir=best_ckpt_dir,
                                                 push_to_hub=push_to_hub,
                                                 hub_model_id=hub_model_id,
                                                 hub_private_repo=hub_private_repo)
            # set False to ensure only best checkpoint is pushed
            argument_kwargs['push_to_hub'] = False
            callbacks = [evaluate_callback]

        CustomTrainer = AngleESETrainer if apply_ese else AngleTrainer
        trainer = CustomTrainer(
            pooler=self.pooler,
            model=self.backbone,
            dataset_format=detect_dataset_format(train_ds),
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
                bf16=bf16,
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_strategy=save_strategy,
                eval_strategy=eval_strategy if valid_ds is not None else 'no',
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
                text_prompt=text_prompt,
                query_prompt=query_prompt,
                doc_prompt=doc_prompt,
                dataset_format=detect_dataset_format(train_ds),
            ),
            model_kwargs=self.model_kwargs,
            **trainer_kwargs
        )
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.backbone = torch.compile(self.backbone)

        trainer.train()
        if argument_kwargs.get('push_to_hub', False):
            trainer.push_to_hub()
        self.backbone.save_pretrained(output_dir)

    def evaluate(self, ds: Dataset, batch_size: int = 32, metric: str = 'spearman_cosine', prompt: Optional[str] = None) -> float:
        """ evaluate

        :param data: Dataset, format A is required
        :param batch_size: int. Default 32.
        :param metric: str. Default 'spearman_cosine'.

        :return: float.
        """
        if prompt is not None:
            ds = ds.map(lambda x: {"text1": prompt.format(text=x["text1"]), "text2": prompt.format(text=x["text2"])}, batched=True)
        return CorrelationEvaluator(
            text1=ds['text1'],
            text2=ds['text2'],
            labels=ds['label'],
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

    @torch.no_grad()
    def encode(self,
               inputs: Union[List[str], Tuple[str], str],
               max_length: Optional[int] = None,
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

        device = device or self.backbone.device
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if prompt is not None:
            inputs = [prompt.format(text=text) for text in inputs]

        tok = self.tokenizer(
            inputs,
            padding=padding,
            max_length=max_length or self.max_length,
            truncation=True,
            return_tensors='pt')
        tok.to(device)

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
