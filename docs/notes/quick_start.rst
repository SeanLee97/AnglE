🚀 Quick Start
================================

A few steps steps to get started with AnglE:


⬇️ Installation
------------------------------------

.. code-block:: bash

    python -m pip install -U angle-emb


Other installation methods, please refer to the `Installation` section.

⌛ Load BERT-based Model
------------------------------------

1) **With Prompts**: You can specify a prompt with `prompt=YOUR_PROMPT` in `encode` method.
If set a prompt, the inputs should be a list of dict or a single dict with key `text`, where `text` is the placeholder in the prompt for the input text. 
You can use other placeholder names. We provide a set of predefined prompts in `Prompts` class, you can check them via `Prompts.list_prompts()`.


.. code-block:: python

    from angle_emb import AnglE, Prompts
    from angle_emb.utils import cosine_similarity


    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    # For retrieval tasks, we use `Prompts.C` as the prompt for the query when using UAE-Large-V1 (no need to specify prompt for documents).
    # When specify prompt, the inputs should be a list of dict with key 'text'
    qv = angle.encode({'text': 'what is the weather?'}, to_numpy=True, prompt=Prompts.C)
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



⌛ Load LLM-based Models
------------------------------------

If the pretrained weight is a LoRA-based model, you need to specify the backbone via `model_name_or_path` and specify the LoRA path via the `pretrained_lora_path` in `from_pretrained` method. 

.. code-block:: python

    from angle_emb import AnglE, Prompts

    angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                                pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
                                pooling_strategy='last',
                                is_llm=True,
                                torch_dtype='float16')

    print('All predefined prompts:', Prompts.list_prompts())
    vec = angle.encode({'text': 'hello world'}, to_numpy=True, prompt=Prompts.A)
    print(vec)
    vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True, prompt=Prompts.A)
    print(vecs)

