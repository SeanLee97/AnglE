# -*- coding: utf-8 -*-

from typing import Union

from .modeling_mistral import BeMistralModel  # NOQA
from .modeling_llama import BeLlamaModel  # NOQA
from .modeling_phi2 import BePhi2Model  # NOQA
from .modeling_opt import BeOPTModel  # NOQA


# register BeLLM models here 
ALL_BELLMS = {
    'BeMistralModel': BeMistralModel,
    'BeLlamaModel': BeLlamaModel,
    'BePhi2Model': BePhi2Model,
    'BeOPTModel': BeOPTModel
}


def check_bellm(model: Union[str, object]) -> bool:
    if isinstance(model, str):
        return model in ALL_BELLMS
    return model in ALL_BELLMS.values()
