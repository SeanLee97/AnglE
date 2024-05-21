# -*- coding: utf-8 -*-

import logging
from typing import List

from scipy import spatial


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AnglE')


def cosine_similarity(vec1: List[int], vec2: List[int]):
    """ Calculate cosine similarity between two vectors.

    :param vec1: a list of integers
    :param vec2: a list of integers
    :return: a float value between 0 and 1, indicating the similarity between the two vectors.
    """
    return 1 - spatial.distance.cosine(vec1, vec2)
