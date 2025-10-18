🚂 Training and Finetuning
============================

There are two types of training methods:

1. use the `angle-trainer` cli to train a model in cli mode.
2. custom training scripts using the `angle` library.


🗂️ Data Preparation
----------------------------------

We currently support three dataset formats:

1. **Format A** (Pair with Label): A pair format with three columns: `text1`, `text2`, and `label`. The `label` should be a similarity score (e.g., 0-1). e.g. `{"text1": "A plane is taking off.", "text2": "An air plane is taking off.",  "label": 0.95}`

2. **Format B** (Query-Positive): A pair format with two columns: `query` and `positive`. Both `query` and `positive` can be either `str` or `List[str]` (if list, one will be randomly sampled during training). e.g. `{"query": "A person on a horse jumps over a broken down airplane.", "positive": "A person is outdoors, on a horse."}`

3. **Format C** (Query-Positive-Negative): A triple format with three columns: `query`, `positive`, and `negative`. All three fields can be either `str` or `List[str]` (if list, one will be randomly sampled during training). e.g. `{"query": "Two blond women are hugging one another.", "positive": "There are women showing affection.", "negative": "Men are fighting."}`

It is required to prepare your data into huggingface `datasets.Dataset` in one of the above formats.


⭐ angle-trainer [recommended]
----------------------------------

You can train a powerful sentence embedding model using the `angle-trainer` cli via a few lines of code.

1. Single gpu training:

    Usage: 

    .. code-block:: bash

        CUDA_VISIBLE_DEVICES=0 angle-trainer --help

2. Multi-gpu training:

    Usage:

    .. code-block:: bash

        CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
        --multi_gpu \
        --num_processes 4 \
        --main_process_port 2345 \
        -m angle_emb.angle_trainer --help


3. Examples:

    a. BERT-based

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



    b. ModernBERT-based

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



    c. LLM-based (Qwen with FSDP)

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



🚂 Custom Train
----------------------------------

You can also train a sentence embedding model using the `angle_emb` library. Here is an example:

.. code-block:: python

    from datasets import load_dataset
    from angle_emb import AnglE


    # 1. load pretrained model
    angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', max_length=128, pooling_strategy='cls').cuda()

    # 2. load dataset
    # `text1`, `text2`, and `label` are three required columns for Format A.
    ds = load_dataset('mteb/stsbenchmark-sts')
    ds = ds.map(lambda obj: {"text1": str(obj["sentence1"]), "text2": str(obj['sentence2']), "label": obj['score']})
    ds = ds.select_columns(["text1", "text2", "label"])

    # 3. fit (no need to tokenize data in advance, it will be done automatically)
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

    # 4. evaluate
    corrcoef = angle.evaluate(ds['test'])
    print('Spearman\'s corrcoef:', corrcoef)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1h28jHvv_x-0fZ0tItIMjf8rJGp3GcO5V?usp=sharing
    :alt: Open In Colab




💡 Hyperparameters
-------------------------

1. `angle_w`: the weight for angle loss. Default `0.02`

2. `ibn_w`: the weight for in-batch negative loss. Default `1.0`

3. `cln_w`: the weight for contrastive learning with hard negative loss. Default `1.0`

4. `cosine_w`: the weight for cosine loss. Default `0.0`

5. `angle_tau`: the temperature for angle loss. Default `20.0`

6. `ibn_tau`: the temperature for ibn and cln losses. Default `20.0`

7. `cosine_tau`: the temperature for cosine loss. Default `20.0`




💡 Fine-tuning Tips
-------------------------

1. If your dataset format is **Format A** (text1, text2, label), it is recommended to slightly increase the weight for `cosine_w` or slightly decrease the weight for `ibn_w`.

2. If your dataset format is **Format B** (query, positive), only `ibn_w` and `ibn_tau` are effective. You don't need to tune other parameters.

3. If your dataset format is **Format C** (query, positive, negative), it is recommended to set `cosine_w` to 0, and set `angle_w` to a small value like 0.02. Be sure to set `cln_w` and `ibn_w`.

4. To alleviate information forgetting in fine-tuning, it is better to specify the `teacher_name_or_path`. If the `teacher_name_or_path` equals `model_name_or_path`, it will conduct self-distillation. **Note that** `teacher_name_or_path` has to have the same tokenizer as `model_name_or_path`. Or it will lead to unexpected results.


💡 Fine-tuning and Infering with `sentence-transformers`
---------------------------------------------------------------------------


1. **Training:** SentenceTransformers also provides a implementation of `AnglE loss <https://sbert.net/docs/package_reference/sentence_transformer/losses.html#angleloss>`_ 
. **But it is partially implemented and may not work well as the official code. We recommend to use the official `angle_emb` for fine-tuning AnglE model.**

2. **Infering:** If your model is trained with `angle_emb`, and you want to use it with `sentence-transformers`.  You can convert it to `sentence-transformers` model using the script `examples/convert_to_sentence_transformers.py <https://github.com/SeanLee97/AnglE/blob/main/scripts/convert_to_sentence_transformer.py>`_.



💡 Others
-------------------------

1. To enable `llm` training, you **must** manually specify `--is_llm 1` and configure appropriate LoRA hyperparameters.
2. To enable `billm` training, please specify `--apply_billm 1` and configure appropriate `billm_model_class` such as `LLamaForCausalLM` (refer to: https://github.com/WhereIsAI/BiLLM?tab=readme-ov-file#usage).
3. To enable espresso sentence embeddings (ESE), please specify `--apply_ese 1` and configure appropriate ESE hyperparameters via `--ese_kl_temperature float` and `--ese_compression_size integer`.
4. To apply prompts during training:
   
   - Use `--text_prompt` for Format A (applies to both text1 and text2)
   - Use `--query_prompt` for query field in Format B/C
   - Use `--doc_prompt` for positive/negative fields in Format B/C

5. To convert the trained AnglE models to `sentence-transformers`, please run `python scripts/convert_to_sentence_transformers.py --help` for more details.
