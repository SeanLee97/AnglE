🚀 Quick Start
================================

Get started with AnglE in just a few minutes!

----

⬇️ Installation
------------------------------------

Install AnglE using uv:

.. code-block:: bash

    uv pip install -U angle-emb


or pip:

.. code-block:: bash

    pip install -U angle-emb


----

🔍 Inference
====================================

1️⃣ BERT-based Models
------------------------------------

**Option A: With Prompts (for Retrieval Tasks)**

Use prompts for retrieval tasks. Prompts should use ``{text}`` as a placeholder. Check available prompts via ``Prompts.list_prompts()``.

.. code-block:: python

    from angle_emb import AnglE, Prompts
    from angle_emb.utils import cosine_similarity

    # Load model
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    
    # Encode query with prompt, documents without prompt
    qv = angle.encode(['what is the weather?'], to_numpy=True, prompt=Prompts.C)
    doc_vecs = angle.encode([
        'The weather is great!',
        'it is rainy today.',
        'i am going to bed'
    ], to_numpy=True)

    # Calculate similarity
    for dv in doc_vecs:
        print(cosine_similarity(qv[0], dv))


**Option B: Without Prompts (for Similarity Tasks)**

For similarity tasks, you can directly encode texts without prompts.

.. code-block:: python

    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity

    # Load model
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    
    # Encode documents
    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ])

    # Calculate pairwise similarity
    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))

----

2️⃣ LLM-based Models
------------------------------------

For LoRA-based models, specify both the backbone model and LoRA weights.

.. note::
   Always set ``is_llm=True`` for LLM models.

.. code-block:: python

    import torch
    from angle_emb import AnglE, Prompts
    from angle_emb.utils import cosine_similarity

    # Load LLM with LoRA weights
    angle = AnglE.from_pretrained(
        'NousResearch/Llama-2-7b-hf',
        pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
        pooling_strategy='last',
        is_llm=True,
        torch_dtype=torch.float16
    ).cuda()

    # Encode with prompt
    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ], prompt=Prompts.A)

    # Calculate similarity
    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))

----

3️⃣ BiLLM-based Models
------------------------------------

Enable bidirectional LLMs with ``apply_billm=True`` and specify the model class.

.. code-block:: python

    import os
    import torch
    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity

    # Set BiLLM environment variable
    os.environ['BiLLM_START_INDEX'] = '31'

    # Load BiLLM model
    angle = AnglE.from_pretrained(
        'NousResearch/Llama-2-7b-hf',
        pretrained_lora_path='SeanLee97/bellm-llama-7b-nli',
        pooling_strategy='last',
        is_llm=True,
        apply_billm=True,
        billm_model_class='LlamaForCausalLM',
        torch_dtype=torch.float16
    ).cuda()

    # Encode with custom prompt
    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ], prompt='The representative word for sentence {text} is:"')

    # Calculate similarity
    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))

----

4️⃣ Espresso/Matryoshka Models
------------------------------------

Truncate layers and embedding dimensions for flexible model compression.

.. code-block:: python

    from angle_emb import AnglE
    from angle_emb.utils import cosine_similarity

    # Load model
    angle = AnglE.from_pretrained('mixedbread-ai/mxbai-embed-2d-large-v1', pooling_strategy='cls').cuda()
    
    # Truncate to specific layer
    angle = angle.truncate_layer(layer_index=22)
    
    # Encode with truncated embedding size
    doc_vecs = angle.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ], embedding_size=768)

    # Calculate similarity
    for i, dv1 in enumerate(doc_vecs):
        for dv2 in doc_vecs[i+1:]:
            print(cosine_similarity(dv1, dv2))

----

⚡ Batch Inference
------------------------------------

Speed up inference with the ``batched`` library for large-scale processing.

**Installation:**

.. code-block:: bash

    python -m pip install batched

**Usage:**

.. code-block:: python

    import batched
    from angle_emb import AnglE

    # Load model
    model = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy='cls').cuda()
    
    # Enable dynamic batching
    model.encode = batched.dynamically(model.encode, batch_size=64)

    # Encode large batch
    vecs = model.encode([
        'The weather is great!',
        'The weather is very good!',
        'i am going to bed'
    ] * 50)

----

📚 Next Steps
------------------------------------

- Learn more about :doc:`training` your own models
- Explore :doc:`pretrained_models` available for use
- Check out the complete :doc:`tutorial` for advanced usage
- Read about :doc:`evaluation` methods
