# -*- coding: utf-8 -*-

from typing import List

import numpy as np
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances
)
from scipy.stats import pearsonr, spearmanr

from .base import AngleBase


class CorrelationEvaluator(object):
    def __init__(
        self,
        text1: List[str],
        text2: List[str],
        labels: List[float],
        batch_size: int = 32
    ):
        assert len(text1) == len(text2) == len(labels), "text1, text2, and labels must have the same length"
        self.text1 = text1
        self.text2 = text2
        self.labels = labels
        self.batch_size = batch_size

    def __call__(self, model: AngleBase, show_progress: bool = True, **kwargs) -> dict:
        """ Evaluate the model on the given dataset.

        :param model: AnglE, the model to evaluate.
        :param show_progress: bool, whether to show a progress bar during evaluation.
        :param kwargs: Additional keyword arguments to pass to the `encode` method of the model.

        :return: dict, The evaluation results.
        """
        embeddings1 = []
        embeddings2 = []
        for chunk in tqdm(chunked_iter(range(len(self.text1)), self.batch_size),
                          total=len(self.text1)//self.batch_size,
                          disable=not show_progress):
            batch_text1 = [self.text1[i] for i in chunk]
            batch_text2 = [self.text2[i] for i in chunk]

            batch_embeddings1 = model.encode(batch_text1, **kwargs)
            batch_embeddings2 = model.encode(batch_text2, **kwargs)
            embeddings1.append(batch_embeddings1)
            embeddings2.append(batch_embeddings2)

        embeddings1 = np.concatenate(embeddings1, axis=0)
        embeddings2 = np.concatenate(embeddings2, axis=0)

        cosine_labels = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        pearson_cosine, _ = pearsonr(self.labels, cosine_labels)
        spearman_cosine, _ = spearmanr(self.labels, cosine_labels)

        pearson_manhattan, _ = pearsonr(self.labels, manhattan_distances)
        spearman_manhattan, _ = spearmanr(self.labels, manhattan_distances)

        pearson_euclidean, _ = pearsonr(self.labels, euclidean_distances)
        spearman_euclidean, _ = spearmanr(self.labels, euclidean_distances)

        pearson_dot, _ = pearsonr(self.labels, dot_products)
        spearman_dot, _ = spearmanr(self.labels, dot_products)

        metrics = {
            "pearson_cosine": pearson_cosine,
            "spearman_cosine": spearman_cosine,
            "pearson_manhattan": pearson_manhattan,
            "spearman_manhattan": spearman_manhattan,
            "pearson_euclidean": pearson_euclidean,
            "spearman_euclidean": spearman_euclidean,
            "pearson_dot": pearson_dot,
            "spearman_dot": spearman_dot,
        }
        return metrics

    def list_all_metrics(self) -> List[str]:
        """ Get a list of all the metrics that can be computed by this evaluator.

        :return: List[str], A list of all the metrics that can be computed by this evaluator.
        """
        return [
            "pearson_cosine",
            "spearman_cosine",
            "pearson_manhattan",
            "spearman_manhattan",
            "pearson_euclidean",
            "spearman_euclidean",
            "pearson_dot",
            "spearman_dot",
        ]
