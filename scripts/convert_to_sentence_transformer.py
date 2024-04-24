# -*- coding: utf-8 -*-

""" This is a script to convert a pre-trained AnglE model to a SentenceTransformer model.
"""

import argparse

from sentence_transformers import models
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True,
                    help='Specify model_name_or_path to set transformer backbone, default roberta-large')
parser.add_argument('--pooling_strategy', type=str, required=True,
                    help='Specify pooling strategy')
parser.add_argument('--max_length', type=int, default=512,
                    help='Specify max length')
parser.add_argument('--push_to_hub', type=int, default=0, choices=[0, 1], help='Specify push_to_hub, default 0')
parser.add_argument('--hub_private_repo', type=int, default=1, choices=[0, 1],
                    help='Specify hub_private_repo, default 1')
parser.add_argument('--hub_model_id', type=str, default=None,
                    help='Specify push_to_hub_model_id, default None, format like organization/model_id')

args = parser.parse_args()

word_embedding_model = models.Transformer(args.model_name_or_path, max_seq_length=args.max_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.pooling_strategy)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

if args.push_to_hub:
    model.push_to_hub(args.hub_model_id, private=args.hub_private_repo, exist_ok=True)
