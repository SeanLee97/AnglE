# -*- coding: utf-8 -*-


def test_eval():
    from datasets import load_dataset
    from angle_emb import AnglE, CorrelationEvaluator

    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls')
    eval_dataset = load_dataset('sentence-transformers/stsb', split="test")

    spearman = CorrelationEvaluator(
        text1=eval_dataset["sentence1"],
        text2=eval_dataset["sentence2"],
        labels=eval_dataset["score"],
    )(angle)['spearman_cosine']
    assert spearman > 0.89

    spearman = angle.evaluate(eval_dataset)
    assert spearman > 0.89
