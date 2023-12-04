# -*- coding: utf-8 -*-

import os
import sys
import json
from pprint import pprint

import jmespath


if len(sys.argv) != 2:
    print('usage: python compute_scores.py result_dir')
    sys.exit()

result_dir = sys.argv[1]


TASK_LIST_CLASSIFICATION = ([
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
], 'Classification', 'test.en.accuracy', {
    'AmazonPolarityClassification': 'test.accuracy',
    'Banking77Classification': 'test.accuracy',
    'EmotionClassification': 'test.accuracy',
    'ImdbClassification': 'test.accuracy',
    'ToxicConversationsClassification': 'test.accuracy',
    'TweetSentimentExtractionClassification': 'test.accuracy',
})

TASK_LIST_CLUSTERING = ([
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
], 'Clustering', 'test.v_measure', {})

TASK_LIST_PAIR_CLASSIFICATION = ([
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
], 'Pair Classification', 'test.cos_sim.ap', {})

TASK_LIST_RERANKING = ([
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
], 'Reranking', 'test.map', {})


TASK_LIST_RETRIEVAL = ([
    "ArguAna",
    "ClimateFEVER",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
], 'Retrieval', 'test.ndcg_at_10', {'MSMARCO': 'dev.ndcg_at_10'})
TASK_LIST_RETRIEVAL2 = ([
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
], 'Retrieval2', 'test.ndcg_at_10', {})
TASK_LIST_STS = ([
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
], 'STS', 'test.cos_sim.spearman', {'STS17': 'test."en-en".cos_sim.spearman', 'STS22': 'test.en.cos_sim.spearman'})

TASK_LIST_SUMM = (['SummEval'], 'Summarization', 'test.cos_sim.spearman', {})


TASKS = [
    TASK_LIST_STS,
    TASK_LIST_SUMM,
    TASK_LIST_CLASSIFICATION,
    TASK_LIST_CLUSTERING,
    TASK_LIST_PAIR_CLASSIFICATION,
    TASK_LIST_RERANKING,
    TASK_LIST_RETRIEVAL,
    TASK_LIST_RETRIEVAL2
]


total_scores = {}
for datasets, name, metric, extra_metric in TASKS:
    scores = []
    for dataset in datasets:
        tmp_metric = metric
        metric = extra_metric.get(dataset, metric)
        with open(os.path.join(result_dir, f'{dataset}.json'), 'r') as reader:
            obj = json.load(reader)
            scores.append(jmespath.search(metric, obj))
        metric = tmp_metric
    print('>>>', list(zip(scores, datasets)))
    if name == 'Retrieval':
        avg = sum(scores)
    else:
        avg = sum(scores) / len(scores)
    print(f'task={name}, avg score={avg}')
    total_scores[name] = avg
total_scores['Retrieval'] = (total_scores['Retrieval'] + total_scores['Retrieval2']) / (len(TASK_LIST_RETRIEVAL[0]) + 1)
del total_scores['Retrieval2']
# del total_scores['Summarization']
pprint(total_scores)

print(f'Total avg: {sum(total_scores.values()) / len(total_scores)}')
