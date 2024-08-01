👨‍🏫 Tutorial
============================


4-steps to train a powerful pubmed sentence embeddings.
------------------------------------------------------------

This tutorial will guide you through the process of training powerful sentence embeddings using PubMed data with the AnglE framework. We'll cover data preparation, model training, and evaluation.


Step 1: Data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Clean data from the `qiaojin/PubMedQA <https://huggingface.co/datasets/qiaojin/PubMedQA>`_ dataset and prepare it into AnglE's `DatasetFormats.C <https://angle.readthedocs.io/en/latest/notes/training.html#data-prepration>`_ format.

We have already processed the data and made it available on HuggingFace: `WhereIsAI/medical-triples <https://huggingface.co/datasets/WhereIsAI/medical-triples/viewer/all_pubmed_en_v1>`_. You can use this processed dataset for this tutorial.


Step 2: Train the model with `angle-trainer`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


To train AnglE embeddings, you'll need to install the `angle-emb` package:

.. code-block:: bash

    python -m pip install -U angle-emb

The `angle-emb` package includes a user-friendly command-line interface called `angle-trainer <https://angle.readthedocs.io/en/latest/notes/training.html#angle-trainer-recommended>`_ for training AnglE embeddings.

With `angle-trainer`, you can quickly start model training by specifying the data path and `hyperparameters <https://angle.readthedocs.io/en/latest/notes/training.html#fine-tuning-tips>`_.

Here's an example of training a BERT-base model:

.. code-block:: bash

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=1234 -m angle_emb.angle_trainer \
    --train_name_or_path WhereIsAI/medical-triples \
    --train_subset_name all_pubmed_en_v1 \
    --save_dir ckpts/pubmedbert-medical-base-v1 \
    --model_name_or_path microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --pooling_strategy cls \
    --maxlen 75 \
    --ibn_w 20.0 \
    --cosine_w 0.0 \
    --angle_w 1.0 \
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


And here's another example of training a BERT-large model:

.. code-block:: bash

    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=1234 -m angle_emb.angle_trainer \
    --train_name_or_path WhereIsAI/medical-triples \
    --train_subset_name all_pubmed_en_v1 \
    --save_dir ckpts/uae-medical-large-v1 \
    --model_name_or_path WhereIsAI/UAE-Large-V1 \
    --load_mlm_model 1 \
    --pooling_strategy cls \
    --maxlen 75 \
    --ibn_w 20.0 \
    --cosine_w 0.0 \
    --angle_w 1.0 \
    --learning_rate 1e-6 \
    --logging_steps 5 \
    --save_steps 500 \
    --warmup_steps 50 \
    --batch_size 32 \
    --seed 42 \
    --gradient_accumulation_steps 3 \
    --push_to_hub 1 --hub_model_id pubmed-angle-large-en --hub_private_repo 1 \
    --epochs 1 \
    --fp16 1


Step 3: Evaluate the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AnglE provides a `CorrelationEvaluator <https://angle.readthedocs.io/en/latest/notes/evaluation.html#spearman-and-pearson-correlation>`_ to evaluate the performance of sentence embeddings.

For convenience, we have processed the `PubMedQA <https://huggingface.co/datasets/qiaojin/PubMedQA/viewer/pqa_labeled>`_ pqa_labeled subset data into the `DatasetFormats.A` format and made it available in `WhereIsAI/pubmedqa-test-angle-format-a <https://huggingface.co/datasets/WhereIsAI/pubmedqa-test-angle-format-a>`_ for evaluation purposes.

The following code demonstrates how to evaluate the trained `pubmed-angle-base-en` model:


.. code-block:: python

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from angle_emb import AnglE, CorrelationEvaluator
    from datasets import load_dataset


    angle = AnglE.from_pretrained('WhereIsAI/pubmed-angle-base-en', pooling_strategy='cls').cuda()

    ds = load_dataset('WhereIsAI/pubmedqa-test-angle-format-a', split='train')

    metric = CorrelationEvaluator(
        text1=ds['text1'],
        text2=ds['text2'],
        labels=ds['label']
    )(angle, show_progress=True)

    print(metric)


Here, we compare the performance of our trained models with two popular models trained on PubMed data. The results are as follows:


+----------------------------------------+-------------------------+
| Model                                  | Spearman's Correlation  |
+========================================+=========================+
| tavakolih/all-MiniLM-L6-v2-pubmed-full | 84.56                   |
+----------------------------------------+-------------------------+
| NeuML/pubmedbert-base-embeddings       | 84.88                   |
+----------------------------------------+-------------------------+
| WhereIsAI/pubmed-angle-base-en         | 86.01                   |
+----------------------------------------+-------------------------+
| WhereIsAI/pubmed-angle-large-en        | **86.21**               |
+----------------------------------------+-------------------------+


The results show that our trained models, `WhereIsAI/pubmed-angle-base-en` and `WhereIsAI/pubmed-angle-large-en`, performs better than other popular models on the PubMedQA dataset.
The large model achieves the highest Spearman's correlation of **86.21**.


Step 4: Use the model in your application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By using `angle-emb`, you can quickly load the model for your applications.

.. code-block:: python

    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity

    angle = AnglE.from_pretrained('WhereIsAI/pubmed-angle-base-en', pooling_strategy='cls').cuda()

    query = 'How to treat childhood obesity and overweight?'
    docs = [
        query,
        'The child is overweight. Parents should relieve their children\'s symptoms through physical activity and healthy eating. First, they can let them do some aerobic exercise, such as jogging, climbing, swimming, etc. In terms of diet, children should eat more cucumbers, carrots, spinach, etc. Parents should also discourage their children from eating fried foods and dried fruits, which are high in calories and fat. Parents should not let their children lie in bed without moving after eating. If their children\'s condition is serious during the treatment of childhood obesity, parents should go to the hospital for treatment under the guidance of a doctor in a timely manner.',
        'If you want to treat tonsillitis better, you can choose some anti-inflammatory drugs under the guidance of a doctor, or use local drugs, such as washing the tonsil crypts, injecting drugs into the tonsils, etc. If your child has a sore throat, you can also give him or her some pain relievers. If your child has a fever, you can give him or her antipyretics. If the condition is serious, seek medical attention as soon as possible. If the medication does not have a good effect and the symptoms recur, the author suggests surgical treatment. Parents should also make sure to keep their children warm to prevent them from catching a cold and getting tonsillitis again.',
    ]

    embeddings = angle.encode(docs)
    query_emb = embeddings[0]

    for doc, emb in zip(docs[1:], embeddings[1:]):
        print(cosine_similarity(query_emb, emb))

    # 0.8029839020052982
    # 0.4260630076818197
