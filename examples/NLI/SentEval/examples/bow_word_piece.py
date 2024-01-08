# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging

from transformers import BertTokenizer

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
a_remove_set = {".", "a", "the", "in", ",", "is", "to", "of", "and", "'", "on", "man", "-", "s", "with", "for", "\"", "at", "##s", "woman", "are", "it", "two", "that", "you", "dog", "said", "playing", "i", "an", "as", "was", "from", ":", "by", "white"}
remove_set = {'?', '*', '#', '´', '’', '=', '…', '|', '~', '/', '‚', '¿', '–', '»', '-', '€', '‘', '"', '(', '•', '`', '$', ':', '[', '”', '%', '£', '<', '[UNK]', ';', '“', '@', '_', '{', '^', ',', '.', '!', '™', '&', ']', '>', '\\', "'", ')', '+', '—'}

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1
        #for word in tokenizer.convert_ids_to_tokens(tokenizer.encode(' '.join(s), add_special_tokens=False)):
            #if '##' in word and word in remove_set: continue
            #words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec

def get_bert_wordvec(path_to_vec, word2id):
    word_vec = {}
    from transformers import BertModel
    bert = BertModel.from_pretrained('bert-base-uncased')
    vocab = tokenizer.get_vocab()
    bert_word_vec = bert.embeddings.word_embeddings.weight.detach().numpy()

    for word in word2id:
        if word in ['<s>', '</s>', '<p>']:
            word_vec[word] = np.zeros(768)
        else:
            word_vec[word] = bert_word_vec[vocab[word]]

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec

# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    #params.word_vec = get_bert_wordvec(PATH_TO_VEC, params.word2id)
    #params.wvec_dim = 768
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        # for word in tokenizer.convert_ids_to_tokens(tokenizer.encode(' '.join(sent), add_special_tokens=False)):
        for word in sent:
            if word in params.word_vec:# and word not in a_remove_set and word not in remove_set:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    #transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      #'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      #'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      #'Length', 'WordContent', 'Depth', 'TopConstituents',
                      #'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      #'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    results = se.eval(transfer_tasks)
    print(results)
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))

    from prettytable import PrettyTable
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)
