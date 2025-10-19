🚂 Training and Finetuning
============================

Train powerful sentence embedding models with AnglE using either CLI or Python API.

----

🗂️ Data Preparation
----------------------------------

AnglE supports three dataset formats. Choose based on your task:

**Format A: Pair with Label**
    A pair format with three columns: ``text1``, ``text2``, and ``label``. 
    The ``label`` should be a similarity score (e.g., 0-1).
    
    Example:
    
    .. code-block:: json
    
        {"text1": "A plane is taking off.", "text2": "An air plane is taking off.", "label": 0.95}

**Format B: Query-Positive**
    A pair format with two columns: ``query`` and ``positive``.
    Both fields can be ``str`` or ``List[str]`` (random sampling for lists).
    
    Example:
    
    .. code-block:: json
    
        {"query": "A person on a horse jumps over a broken down airplane.", "positive": "A person is outdoors, on a horse."}

**Format C: Query-Positive-Negative**
    A triple format with three columns: ``query``, ``positive``, and ``negative``.
    All fields can be ``str`` or ``List[str]`` (random sampling for lists).
    
    Example:
    
    .. code-block:: json
    
        {"query": "Two blond women are hugging one another.", "positive": "There are women showing affection.", "negative": "Men are fighting."}

.. note::
   All formats use HuggingFace ``datasets.Dataset``.

----

🎯 Training Methods
----------------------------------

⭐ Method 1: CLI Training (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``angle-trainer`` to train your models with a simple command-line interface.

**Single GPU Training:**

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 angle-trainer --help

**Multi-GPU Training:**

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --main_process_port 2345 \
    -m angle_emb.angle_trainer --help


use FSDP for bigger batch size:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --main_process_port 2345 \
    --config_file examples/FSDP/fsdp_config.yaml \
    -m angle_emb.angle_trainer \
    --gradient_checkpointing 1 \
    --use_reentrant 0 \
    ...

see more examples in `examples/FSDP <https://github.com/SeanLee97/AnglE/tree/main/examples/FSDP>`_

📝 Training Examples
""""""""""""""""""""""""""""""""""""""""""""

**Example 1: BERT-based Model**

Train a BERT model with multi-GPU support:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --main_process_port 2345 \
    -m angle_emb.angle_trainer \
    --model_name_or_path WhereIsAI/UAE-Large-V1 \
    --train_name_or_path SeanLee97/nli_for_simcse \
    --save_dir ckpts/uae-nli \
    --column_rename_mapping "text:query" \
    --query_prompt "query: {text}" \
    --doc_prompt "doc: {text}" \
    --learning_rate 1e-5 \
    --pooling_strategy cls \
    --epochs 1 \
    --batch_size 32 \
    --logging_steps 10 \
    --gradient_accumulation_steps 2 \
    --ibn_w 1.0 \
    --cln_w 1.0 \
    --angle_w 0.02 \
    --fp16 1

**Example 2: ModernBERT-based Model**

Train with ModernBERT architecture:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --main_process_port 2345 \
    -m angle_emb.angle_trainer \
    --model_name_or_path answerdotai/ModernBERT-base \
    --train_name_or_path SeanLee97/nli_for_simcse \
    --save_dir ckpts/modernbert-nli \
    --column_rename_mapping "text:query" \
    --query_prompt "query: {text}" \
    --doc_prompt "doc: {text}" \
    --learning_rate 1e-4 \
    --pooling_strategy mean \
    --epochs 1 \
    --batch_size 128 \
    --logging_steps 10 \
    --gradient_accumulation_steps 2 \
    --ibn_w 1.0 \
    --cln_w 1.0 \
    --angle_w 0.02 \
    --fp16 1

**Example 3: LLM-based Model with FSDP**

Train large language models using Fully Sharded Data Parallel:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --main_process_port 2345 \
    --config_file examples/FSDP/fsdp_config.yaml \
    -m angle_emb.angle_trainer \
    --gradient_checkpointing 1 \
    --use_reentrant 0 \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --torch_dtype "bfloat16" \
    --is_llm 1 \
    --apply_lora 1 --lora_r 32 --lora_alpha 32 \
    --maxlen 312 \
    --train_name_or_path SeanLee97/nli_for_simcse \
    --save_dir ckpts/qwen-nli \
    --column_rename_mapping "text:query" \
    --query_prompt "query: {text}" \
    --doc_prompt "doc: {text}" \
    --learning_rate 1e-4 \
    --pooling_strategy last \
    --epochs 1 \
    --batch_size 16 \
    --logging_steps 10 \
    --gradient_accumulation_steps 2 \
    --ibn_w 1.0 \
    --cln_w 1.0 \
    --angle_w 0.02 \
    --bf16 1

🐍 Method 2: Python API Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train models programmatically using the ``angle_emb`` library.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1h28jHvv_x-0fZ0tItIMjf8rJGp3GcO5V?usp=sharing
    :alt: Open In Colab

**Example:**

.. code-block:: python

    from datasets import load_dataset
    from angle_emb import AnglE

    # Step 1: Load pretrained model
    angle = AnglE.from_pretrained(
        'SeanLee97/angle-bert-base-uncased-nli-en-v1',
        max_length=128,
        pooling_strategy='cls'
    ).cuda()

    # Step 2: Prepare dataset (Format A example)
    ds = load_dataset('mteb/stsbenchmark-sts')
    ds = ds.map(lambda obj: {
        "text1": str(obj["sentence1"]),
        "text2": str(obj['sentence2']),
        "label": obj['score']
    })
    ds = ds.select_columns(["text1", "text2", "label"])

    # Step 3: Train the model
    angle.fit(
        train_ds=ds['train'].shuffle(),
        valid_ds=ds['validation'],
        output_dir='ckpts/sts-b',
        batch_size=32,
        epochs=5,
        learning_rate=2e-5,
        save_steps=100,
        eval_steps=1000,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        loss_kwargs={
            'cosine_w': 1.0,
            'ibn_w': 1.0,
            'angle_w': 0.02,
            'cosine_tau': 20,
            'ibn_tau': 20,
            'angle_tau': 20
        },
        fp16=True,
        logging_steps=100
    )

    # Step 4: Evaluate
    corrcoef = angle.evaluate(ds['test'])
    print('Spearman\'s corrcoef:', corrcoef)

----

⚙️ Configuration & Hyperparameters
----------------------------------

💡 Loss Weight Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------+-------------------+---------------------------------------+
| Parameter      | Default Value     | Description                           |
+================+===================+=======================================+
| ``angle_w``    | 0.02              | Weight for angle loss                 |
+----------------+-------------------+---------------------------------------+
| ``ibn_w``      | 1.0               | Weight for in-batch negative loss     |
+----------------+-------------------+---------------------------------------+
| ``cln_w``      | 1.0               | Weight for contrastive learning loss  |
+----------------+-------------------+---------------------------------------+
| ``cosine_w``   | 0.0               | Weight for cosine loss                |
+----------------+-------------------+---------------------------------------+

💡 Temperature Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------+-------------------+---------------------------------------+
| Parameter      | Default Value     | Description                           |
+================+===================+=======================================+
| ``angle_tau``  | 20.0              | Temperature for angle loss            |
+----------------+-------------------+---------------------------------------+
| ``ibn_tau``    | 20.0              | Temperature for ibn and cln losses    |
+----------------+-------------------+---------------------------------------+
| ``cosine_tau`` | 20.0              | Temperature for cosine loss           |
+----------------+-------------------+---------------------------------------+

💡 Fine-tuning Tips
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Format-specific Recommendations:**

+-------------+---------------------------------------------------------------+
| Format      | Recommendation                                                |
+=============+===============================================================+
| **Format A**| Increase ``cosine_w`` or decrease ``ibn_w``                  |
+-------------+---------------------------------------------------------------+
| **Format B**| Only tune ``ibn_w`` and ``ibn_tau``                          |
+-------------+---------------------------------------------------------------+
| **Format C**| Set ``cosine_w=0``, ``angle_w=0.02``, configure ``cln_w``    |
+-------------+---------------------------------------------------------------+

**Prevent Catastrophic Forgetting:**

To alleviate information forgetting during fine-tuning:

- Set ``teacher_name_or_path`` for knowledge distillation
- Use same model path for self-distillation
- **Important:** Teacher and student must use the **same tokenizer**

⚙️ Advanced Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Training Special Models:**

+------------------+-------------------------------------------------------------+
| Model Type       | CLI Flags                                                   |
+==================+=============================================================+
| **LLM**          | ``--is_llm 1`` + LoRA parameters                            |
+------------------+-------------------------------------------------------------+
| **BiLLM**        | ``--apply_billm 1 --billm_model_class LlamaForCausalLM``   |
+------------------+-------------------------------------------------------------+
| **Espresso**     | ``--apply_ese 1 --ese_kl_temperature 1.0``                  |
+------------------+-------------------------------------------------------------+

**Applying Prompts:**

+-------------+-------------------------+-----------------------------------+
| Format      | Flag                    | Applies To                        |
+=============+=========================+===================================+
| Format A    | ``--text_prompt``       | Both ``text1`` and ``text2``      |
+-------------+-------------------------+-----------------------------------+
| Format B/C  | ``--query_prompt``      | ``query`` field                   |
+-------------+-------------------------+-----------------------------------+
| Format B/C  | ``--doc_prompt``        | ``positive`` and ``negative``     |
+-------------+-------------------------+-----------------------------------+

**Model Conversion:**

Convert trained models to ``sentence-transformers`` format:

.. code-block:: bash

    python scripts/convert_to_sentence_transformers.py --help

🔄 Integration with sentence-transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Training:**

SentenceTransformers provides an `AnglE loss <https://sbert.net/docs/package_reference/sentence_transformer/losses.html#angleloss>`_ implementation.

.. warning::
   The SentenceTransformers implementation is partial. For best results, use the official ``angle_emb`` library.

**Inference:**

Models trained with ``angle_emb`` can be converted to ``sentence-transformers`` format using the conversion script at ``examples/convert_to_sentence_transformers.py``.

📚 Additional Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Check out the complete :doc:`tutorial` for a hands-on example
- Learn about :doc:`evaluation` methods
- Explore available :doc:`pretrained_models`
- See :doc:`quickstart` for basic usage
