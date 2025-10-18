🚀 Quick Start
================================

A few steps to get started with AnglE:


⬇️ Installation
------------------------------------

.. code-block:: bash

    python -m pip install -U angle-emb


Other installation methods, please refer to the `Installation` section.

⌛ Infer BERT-based Model
------------------------------------

1) **With Prompts**: You can specify a prompt with `prompt=YOUR_PROMPT` in `encode` method. The prompt should use `{text}` as the placeholder. We provide a set of predefined prompts in `Prompts` class, you can check them via `Prompts.list_prompts()`.


.. code-block:: python

    from angle_emb import AnglE, Prompts
    from angle_emb.utils import cosine_similarity


    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    # For retrieval tasks, we use `Prompts.C` as the prompt for the query when using UAE-Large-V1 (no need to specify prompt for documents).
    qv = angle.encode(['what is the weather?'], to_numpy=True, prompt=Prompts.C)
    doc_vecs = angle.encode([
        'The weather is great!',
        'it is rainy today.',
        'i am going to bed'
    ], to_numpy=True)

    for dv in doc_vecs:
        print(cosine_similarity(qv[0], dv))


2) **Without Prompts**: no need to specify a prompt. Just input a list of strings or a single string.


.. code-block:: python

    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity


    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    # for non-retrieval tasks, we don't need to specify prompt when using UAE-Large-V1.
    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ])

    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))



⌛ Infer LLM-based Models
------------------------------------

If the pretrained weight is a LoRA-based model, you need to specify the backbone via `model_name_or_path` and specify the LoRA path via the `pretrained_lora_path` in `from_pretrained` method. **You must manually set `is_llm=True`** for LLM models.

.. code-block:: python

    import torch
    from angle_emb import AnglE, Prompts
    from angle_emb.utils import cosine_similarity

    angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                                  pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
                                  pooling_strategy='last',
                                  is_llm=True,
                                  torch_dtype=torch.float16).cuda()
    print('All predefined prompts:', Prompts.list_prompts())
    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ], prompt=Prompts.A)

    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))



⌛ Infer BiLLM-based Models
------------------------------------

Specify `apply_billm` and `billm_model_class` to load and infer billm models

.. code-block:: python

    import os
    # set an environment variable for billm start index
    os.environ['BiLLM_START_INDEX'] = '31'

    import torch
    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity

    # specify `apply_billm` and `billm_model_class` to load billm models
    # You must manually set is_llm=True for LLM models
    angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                                  pretrained_lora_path='SeanLee97/bellm-llama-7b-nli',
                                  pooling_strategy='last',
                                  is_llm=True,
                                  apply_billm=True,
                                  billm_model_class='LlamaForCausalLM',
                                  torch_dtype=torch.float16).cuda()

    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ], prompt='The representative word for sentence {text} is:"')

    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))



⌛ Infer Espresso/Matryoshka Models
------------------------------------

Specify `layer_index` and `embedding_size` to truncate embeddings.

.. code-block:: python

    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity


    angle = AnglE.from_pretrained('mixedbread-ai/mxbai-embed-2d-large-v1', pooling_strategy='cls').cuda()
   # truncate layer
    angle = angle.truncate_layer(layer_index=22)
    # specify embedding size to truncate embeddings
    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ], embedding_size=768)

    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))



⌛ Batch Inference
------------------------------------

It is recommended to use Mixedbread's `batched` library to speed up the inference process.


.. code-block:: bash

    python -m pip install batched



.. code-block:: python

    import batched
    from angle_emb import AnglE

    model = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy='cls').cuda()
    model.encode = batched.dynamically(model.encode, batch_size=64)

    vecs = model.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ] * 50)

