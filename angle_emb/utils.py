import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from scipy import spatial
from transformers import PreTrainedModel

logger = logging.getLogger('AnglE')
logger.setLevel(logging.INFO)


def cosine_similarity(vec1: List[int], vec2: List[int]):
    """ Calculate cosine similarity between two vectors.

    :param vec1: a list of integers
    :param vec2: a list of integers
    :return: a float value between 0 and 1, indicating the similarity between the two vectors.
    """
    return 1 - spatial.distance.cosine(vec1, vec2)


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
            outputs * inputs["attention_mask"][:, :, None], dim=1) / inputs["attention_mask"].sum(dim=1).unsqueeze(1)
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
