👨‍🏫 Tutorial
============================

A complete walkthrough: Train powerful sentence embeddings for medical text.

This tutorial demonstrates how to train domain-specific sentence embeddings using PubMed data with the AnglE framework. You'll learn data preparation, model training, evaluation, and practical application.

----

📋 Overview
----------------------------------

In this tutorial, you will:

1. 📦 Prepare a medical text dataset from PubMed
2. 🚂 Train a sentence embedding model
3. 📊 Evaluate the model performance
4. 🔧 Apply the model in practice

**Expected Time:** 2-4 hours (depending on hardware)

**Prerequisites:** 
- Python 3.7+
- CUDA-compatible GPU(s)
- Basic knowledge of PyTorch and HuggingFace

----

Step 1: Data Preparation
====================================

📥 Dataset Selection
----------------------------------

We'll use the `PubMedQA <https://huggingface.co/datasets/qiaojin/PubMedQA>`_ dataset for training.

**Pre-processed Dataset Available:**

For convenience, we've already processed the data into AnglE's **Format C** (query, positive, negative) and made it available on HuggingFace:

📦 `WhereIsAI/medical-triples <https://huggingface.co/datasets/WhereIsAI/medical-triples/viewer/all_pubmed_en_v1>`_

.. note::
   Format C is ideal for contrastive learning with hard negatives. See :doc:`training` for more format options.

----

Step 2: Train the Model
====================================

⬇️ Installation
----------------------------------

First, install the ``angle-emb`` library:

.. code-block:: bash

    python -m pip install -U angle-emb

🎯 Training with angle-trainer
----------------------------------

Use the ``angle-trainer`` CLI for streamlined training. You'll need to specify:

- Dataset path and hyperparameters
- Model architecture
- Training configuration

See :doc:`training` for detailed parameter descriptions.

----

📝 Training Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example 1: Train BERT-base Model**

Train a base model suitable for general medical text:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1,2,3 WANDB_MODE=disabled accelerate launch \
    --multi_gpu \
    --num_processes 3 \
    --main_process_port 1234 \
    -m angle_emb.angle_trainer \
    --train_name_or_path WhereIsAI/medical-triples \
    --train_subset_name all_pubmed_en_v1 \
    --save_dir ckpts/pubmedbert-medical-base-v1 \
    --model_name_or_path microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --pooling_strategy cls \
    --maxlen 75 \
    --ibn_w 1.0 \
    --cln_w 1.0 \
    --cosine_w 0.0 \
    --angle_w 0.02 \
    --learning_rate 1e-6 \
    --logging_steps 5 \
    --save_steps 500 \
    --warmup_steps 50 \
    --batch_size 64 \
    --seed 42 \
    --gradient_accumulation_steps 3 \
    --push_to_hub 1 --hub_model_id pubmed-angle-base-en --hub_private_repo 1 \
    --epochs 1 \
    --fp16 1

**Key Parameters Explained:**

- ``--model_name_or_path``: Pre-trained model specialized for biomedical text
- ``--ibn_w``, ``--cln_w``, ``--angle_w``: Loss weights for Format C
- ``--maxlen 75``: Sequence length optimized for PubMed abstracts
- ``--push_to_hub 1``: Automatically upload to HuggingFace Hub

----

**Example 2: Train BERT-large Model**

Train a larger model for better performance:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1,2,3 WANDB_MODE=disabled accelerate launch \
    --multi_gpu \
    --num_processes 3 \
    --main_process_port 1234 \
    -m angle_emb.angle_trainer \
    --train_name_or_path WhereIsAI/medical-triples \
    --column_rename_mapping "text:query" \
    --train_subset_name all_pubmed_en_v1 \
    --save_dir ckpts/uae-medical-large-v1 \
    --model_name_or_path WhereIsAI/UAE-Large-V1 \
    --pooling_strategy cls \
    --maxlen 75 \
    --ibn_w 1.0 \
    --cln_w 1.0 \
    --cosine_w 0.0 \
    --angle_w 0.02 \
    --learning_rate 1e-6 \
    --logging_steps 5 \
    --save_steps 500 \
    --warmup_steps 50 \
    --batch_size 32 \
    --seed 42 \
    --gradient_accumulation_steps 2 \
    --push_to_hub 1 --hub_model_id pubmed-angle-large-en --hub_private_repo 1 \
    --epochs 1 \
    --fp16 1

.. tip::
   Fine-tuning from a general-purpose model (like UAE-Large-V1) often yields better results than training from scratch.

----

Step 3: Evaluate the Model
====================================

📊 Evaluation Setup
----------------------------------

AnglE provides a ``CorrelationEvaluator`` to measure embedding quality using Spearman's correlation.

**Evaluation Dataset:**

We've prepared the `PubMedQA <https://huggingface.co/datasets/qiaojin/PubMedQA/viewer/pqa_labeled>`_ test set in **Format A** (text1, text2, label):

📦 `WhereIsAI/pubmedqa-test-angle-format-a <https://huggingface.co/datasets/WhereIsAI/pubmedqa-test-angle-format-a>`_

----

📈 Evaluation Code
----------------------------------

Evaluate your trained model:

.. code-block:: python

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from angle_emb import AnglE, CorrelationEvaluator
    from datasets import load_dataset

    # Load trained model
    angle = AnglE.from_pretrained(
        'WhereIsAI/pubmed-angle-base-en',
        pooling_strategy='cls'
    ).cuda()

    # Load evaluation dataset
    ds = load_dataset('WhereIsAI/pubmedqa-test-angle-format-a', split='train')

    # Evaluate
    metric = CorrelationEvaluator(
        text1=ds['text1'],
        text2=ds['text2'],
        labels=ds['label']
    )(angle, show_progress=True)

    print(metric)

----

📊 Benchmark Results
----------------------------------

Comparison of models trained on PubMed data:

+------------------------------------------+----------------------------+
| Model                                    | Spearman's Correlation     |
+==========================================+============================+
| tavakolih/all-MiniLM-L6-v2-pubmed-full   | 84.56                      |
+------------------------------------------+----------------------------+
| NeuML/pubmedbert-base-embeddings         | 84.88                      |
+------------------------------------------+----------------------------+
| WhereIsAI/pubmed-angle-base-en           | 86.01                      |
+------------------------------------------+----------------------------+
| **WhereIsAI/pubmed-angle-large-en**      | **86.21** 🏆               |
+------------------------------------------+----------------------------+

.. note::
   The AnglE-trained models outperform existing popular models, with the large variant achieving the highest correlation of **86.21**.

----

Step 4: Use the Model
====================================

🔧 Practical Application
----------------------------------

Load and use your trained model for semantic similarity tasks:

.. code-block:: python

    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity

    # Load model
    angle = AnglE.from_pretrained(
        'WhereIsAI/pubmed-angle-base-en',
        pooling_strategy='cls'
    ).cuda()

    # Define query and documents
    query = 'How to treat childhood obesity and overweight?'
    docs = [
        query,
        'The child is overweight. Parents should relieve their children\'s '
        'symptoms through physical activity and healthy eating. First, they '
        'can let them do some aerobic exercise, such as jogging, climbing, '
        'swimming, etc. In terms of diet, children should eat more cucumbers, '
        'carrots, spinach, etc. Parents should also discourage their children '
        'from eating fried foods and dried fruits, which are high in calories '
        'and fat. Parents should not let their children lie in bed without '
        'moving after eating. If their children\'s condition is serious during '
        'the treatment of childhood obesity, parents should go to the hospital '
        'for treatment under the guidance of a doctor in a timely manner.',
        'If you want to treat tonsillitis better, you can choose some '
        'anti-inflammatory drugs under the guidance of a doctor, or use local '
        'drugs, such as washing the tonsil crypts, injecting drugs into the '
        'tonsils, etc. If your child has a sore throat, you can also give him '
        'or her some pain relievers. If your child has a fever, you can give '
        'him or her antipyretics. If the condition is serious, seek medical '
        'attention as soon as possible. If the medication does not have a good '
        'effect and the symptoms recur, the author suggests surgical treatment. '
        'Parents should also make sure to keep their children warm to prevent '
        'them from catching a cold and getting tonsillitis again.',
    ]

    # Encode all texts
    embeddings = angle.encode(docs)
    query_emb = embeddings[0]

    # Calculate similarities
    for doc, emb in zip(docs[1:], embeddings[1:]):
        similarity = cosine_similarity(query_emb, emb)
        print(f"Similarity: {similarity:.4f}")

**Output:**

.. code-block:: text

    Similarity: 0.8030  # Highly relevant (obesity treatment)
    Similarity: 0.4261  # Less relevant (tonsillitis treatment)

.. tip::
   Higher similarity scores indicate more relevant documents. Use this for search, ranking, or clustering tasks.

----

🎓 Summary
====================================

Congratulations! You've learned how to:

✅ Prepare domain-specific datasets for sentence embedding training

✅ Train BERT-based models using the ``angle-trainer`` CLI

✅ Evaluate model performance with correlation metrics

✅ Apply trained models for semantic similarity tasks

----

📚 Next Steps
----------------------------------

- Explore :doc:`training` for advanced configuration options
- Learn about different :doc:`evaluation` methods
- Check out :doc:`pretrained_models` for ready-to-use models
- Return to :doc:`quickstart` for basic inference examples

**Questions?** See :doc:`citation` for how to cite this work in your research.
