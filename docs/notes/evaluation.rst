🎯 Evaluation
============================


🎯 Spearman and Pearson Correlation
-----------------------------------------

Spearman's and Pearson's correlation coefficients are commonly used to evaluate text embedding quality.

We provide two ways to evaluate the text embeddings by Spearman and Pearson Correlation

1) use `angle.evaluate(dataset)` function


.. code-block:: python

    from angle_emb import AnglE
    from datasets import load_dataset


    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1').cuda()


    ds = load_dataset('mteb/stsbenchmark-sts', split='test')
    ds = ds.map(lambda obj: {"text1": str(obj["sentence1"]), "text2": str(obj['sentence2']), "label": obj['score']})
    ds = ds.select_columns(["text1", "text2", "label"])

    angle.evaluate(ds, metric='spearman_cosine')



2) use `angle_emb.CorrelationEvaluator` evaluator


.. code-block:: python

    from angle_emb import AnglE, CorrelationEvaluator
    from datasets import load_dataset


    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1').cuda()

    ds = load_dataset('mteb/stsbenchmark-sts', split='test')
    ds = ds.map(lambda obj: {"text1": str(obj["sentence1"]), "text2": str(obj['sentence2']), "label": obj['score']})
    ds = ds.select_columns(["text1", "text2", "label"])

    metric = CorrelationEvaluator(
        text1=ds['text1'],
        text2=ds['text2'],
        labels=ds['label']
    )(angle)

    print(metric)

