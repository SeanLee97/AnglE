🎯 Evaluation
============================

Measure the quality of your sentence embeddings using correlation metrics.

----

📊 Overview
----------------------------------

AnglE provides evaluation tools to assess embedding quality using:

- **Spearman's Correlation**: Measures monotonic relationships
- **Pearson's Correlation**: Measures linear relationships

These metrics compare predicted similarities against ground truth labels, commonly used in semantic textual similarity (STS) tasks.

----

🎯 Spearman and Pearson Correlation
----------------------------------

Two methods are available for evaluation:

Method 1: Using angle.evaluate()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to evaluate your model on a dataset.

**Example:**

.. code-block:: python

    from angle_emb import AnglE
    from datasets import load_dataset

    # Load model
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1').cuda()

    # Load and prepare dataset (Format A: text1, text2, label)
    ds = load_dataset('mteb/stsbenchmark-sts', split='test')
    ds = ds.map(lambda obj: {
        "text1": str(obj["sentence1"]),
        "text2": str(obj['sentence2']),
        "label": obj['score']
    })
    ds = ds.select_columns(["text1", "text2", "label"])

    # Evaluate with Spearman correlation
    score = angle.evaluate(ds, metric='spearman_cosine')
    print(f"Spearman's correlation: {score:.4f}")

**Available Metrics:**

- ``spearman_cosine``: Spearman correlation with cosine similarity
- ``pearson_cosine``: Pearson correlation with cosine similarity

Method 2: Using CorrelationEvaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

More flexible evaluation with explicit control over inputs.

**Example:**

.. code-block:: python

    from angle_emb import AnglE, CorrelationEvaluator
    from datasets import load_dataset

    # Load model
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1').cuda()

    # Load and prepare dataset
    ds = load_dataset('mteb/stsbenchmark-sts', split='test')
    ds = ds.map(lambda obj: {
        "text1": str(obj["sentence1"]),
        "text2": str(obj['sentence2']),
        "label": obj['score']
    })
    ds = ds.select_columns(["text1", "text2", "label"])

    # Create evaluator and run
    metric = CorrelationEvaluator(
        text1=ds['text1'],
        text2=ds['text2'],
        labels=ds['label']
    )(angle, show_progress=True)

    print(metric)

**Output Format:**

.. code-block:: python

    {
        'spearman_cosine': 0.8521,
        'pearson_cosine': 0.8432
    }

----

📚 Next Steps
----------------------------------

- Learn how to :doc:`training` models for better performance
- Follow the complete :doc:`tutorial` for hands-on practice
- Check :doc:`quickstart` for basic inference
- Explore :doc:`pretrained_models` for ready-to-use models
