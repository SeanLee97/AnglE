# modified from: https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/eval_MTEB.py

import argparse
import random

from emb_model import EmbModel
from mteb import MTEB


RETRIEVAL_INSTRUCT = 'Represent this sentence for searching relevant passages:'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default=None, required=True, type=str)
    parser.add_argument('--angle_name_or_path', default=None, required=True, type=str)
    parser.add_argument('--task_type', default=None, type=str, help="task type. Default is None, which means using all task types")
    parser.add_argument('--add_instruction', action='store_true', help="whether to add instruction for query")
    parser.add_argument('--pooling_method', default='cls', type=str)
    parser.add_argument('--batch_size', default=300, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = EmbModel(model_name_or_path=args.model_name_or_path,
                     angle_name_or_path=args.angle_name_or_path,
                     normalize_embeddings=False,  # normlize embedding will harm the performance of classification task
                     query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                     pooling_method=args.pooling_method,
                     batch_size=args.batch_size)

    
    task_names = [
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
        "SummEval",
    ] + ['TRECCOVID']
    task_names = [t.description["name"] for t in MTEB(task_types=args.task_type, task_langs=['en']).tasks]
    random.shuffle(task_names)
    for task in task_names:
        if task in ['MSMARCOv2']:
            print('Skip task: {}, since it has no test split'.format(task))
            continue

        if 'CQADupstack' in task or task in ['Touche2020', 'SciFact', 'TRECCOVID', 'NQ',
                                             'NFCorpus', 'MSMARCO', 'HotpotQA', 'FiQA2018',
                                             'FEVER', 'DBPedia', 'ClimateFEVER', 'SCIDOCS', ]:
            instruction = RETRIEVAL_INSTRUCT
        else:
            instruction = None

        model.query_instruction_for_retrieval = instruction

        evaluation = MTEB(tasks=[task], task_langs=['en'], eval_splits = ["test" if task not in ['MSMARCO'] else 'dev'])
        evaluation.run(model, output_folder=f"en_results/{args.angle_name_or_path.split('/')[-1]}")
