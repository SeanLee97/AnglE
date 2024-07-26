🚂 Training and Finetuning
============================

There are two types of training methods:

1. use the `angle-trainer` cli to train a model in cli mode.
2. custom training scripts using the `angle` library.


🗂️ Data Prepration
----------------------------------

We currently support three dataset formats:

1. `DatasetFormats.A`: it is a pair format with three columns: `text1`, `text2`, and `label` (0/1). e.g. `{"text1": "A plane is taking off.", "text2": "An air plane is taking off.",  "label": 1}`

2. `DatasetFormats.B`: it is a triple format with three columns: `text`, `positive`, and `negative`. `positive` and `negative` are the positive and negative samples of `text`. e.g. `{"text": "A person on a horse jumps over a broken down airplane.", "positive": "A person is outdoors, on a horse.", "negative": "A person is at a diner, ordering an omelette."}`

3. `DatasetFormats.C`: it is a pair format with two columns: `text`, `positive`. `positive` is the positive sample of `text`. e.g.  `{"text": "Two blond women are hugging one another.", "positive": "There are women showing affection."}`

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

        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 -m angle_emb.angle_trainer --help


3. Examples:

    a. BERT-based

    .. code-block:: bash

        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 -m angle_emb.angle_trainer \
        --train_name_or_path SeanLee97/all_nli_angle_format_a \
        --save_dir ckpts/bert-base-nli-test \
        --model_name_or_path google-bert/bert-base-uncased \
        --pooling_strategy cls \
        --maxlen 128 \
        --ibn_w 30.0 \
        --cosine_w 0.0 \
        --angle_w 1.0 \
        --angle_tau 20.0 \
        --learning_rate 5e-5 \
        --push_to_hub 1 --hub_model_id SeanLee97/bert-base-nli-test-0728 --hub_private_repo 1 \
        --logging_steps 10 \
        --save_steps 100 \
        --warmup_steps 50 \
        --batch_size 128 \
        --seed 42 \
        --gradient_accumulation_steps 16 \
        --epochs 10 \
        --fp16 1



    b. LLaMA-based

    .. code-block:: bash

        BiLLM_START_INDEX=0 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=2345 -m angle_emb.angle_trainer \
        --train_name_or_path SeanLee97/all_nli_angle_format_b \
        --save_dir ckpts/billm-llama7b-nli \
        --model_name_or_path NousResearch/Llama-2-7b-chat-hf \
        --pooling_strategy avg \
        --maxlen 60 \
        --ibn_w 20.0 \
        --cosine_w 0.0 \
        --angle_w 1.0 \
        --learning_rate 2e-4 \
        --apply_lora 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.1 \
        --load_kbit 4 \
        --is_llm 1 \
        --apply_billm 1 \
        --billm_model_class LlamaForCausalLM \
        --push_to_hub 1 --hub_model_id SeanLee97/test-billm-llama7b-nli --hub_private_repo 1 \
        --logging_steps 5 \
        --save_steps 50 \
        --warmup_steps 50 \
        --batch_size 120 \
        --gradient_accumulation_steps 32 \
        --epochs 2 \
        --fp16 1


🚂 Custom Train
----------------------------------

You can also train a sentence embedding model using the `angle_emb` library. Here is an example:

.. code-block:: python

    from datasets import load_dataset
    from angle_emb import AnglE, AngleDataTokenizer


    # 1. load pretrained model
    angle = AnglE.from_pretrained('SeanLee97/angle-bert-base-uncased-nli-en-v1', max_length=128, pooling_strategy='cls').cuda()

    # 2. load dataset
    # `text1`, `text2`, and `label` are three required columns.
    ds = load_dataset('mteb/stsbenchmark-sts')
    ds = ds.map(lambda obj: {"text1": str(obj["sentence1"]), "text2": str(obj['sentence2']), "label": obj['score']})
    ds = ds.select_columns(["text1", "text2", "label"])

    # 3. transform data
    train_ds = ds['train'].shuffle().map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
    valid_ds = ds['validation'].map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
    test_ds = ds['test'].map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)

    # 4. fit
    angle.fit(
        train_ds=train_ds,
        valid_ds=valid_ds,
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
            'angle_w': 1.0,
            'cosine_tau': 20,
            'ibn_tau': 20,
            'angle_tau': 20
        },
        fp16=True,
        logging_steps=100
    )

    # 5. evaluate
    corrcoef, accuracy = angle.evaluate(test_ds, device=angle.device)
    print('corrcoef:', corrcoef)


.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/1h28jHvv_x-0fZ0tItIMjf8rJGp3GcO5V?usp=sharing
    :alt: Open In Colab


💡 Fine-tuning Tips
-------------------------

1. If your dataset format is `DatasetFormats.A`, it is recommended to slightly increase the weight for `cosine_w` or slightly decrease the weight for `ibn_w`.

2. If your dataset format is `DatasetFormats.B`, it is recommended to set `cosine_w` to 0, and increase the weight for `ibn_w` such as 10 and 20. The `angle_tau` is recommended to set to 20.0.

3. If your dataset format is `DatasetFormats.C`, only `ibn_w` and `ibn_tau` are effective. You don't need to tune other parameters.

4. To alleviate information forgetting in fine-tuning, it is better to specify the `teacher_name_or_path`. If the `teacher_name_or_path` equals `model_name_or_path`, it will conduct self-distillation. **Note that** `teacher_name_or_path` has to have the same tokenizer as `model_name_or_path`. Or it will lead to unexpected results.



💡 Others
-------------------------

1. To enable `llm` training, please specify `--is_llm 1` and configure appropriate LoRA hyperparameters.
2. To enable `billm` training, please specify `--apply_billm 1` and configure appropriate `billm_model_class` such as `LLamaForCausalLM` (refer to: https://github.com/WhereIsAI/BiLLM?tab=readme-ov-file#usage).
3. To enable espresso sentence embeddings (ESE), please specify `--apply_ese 1` and configure appropriate ESE hyperparameters via `--ese_kl_temperature float` and `--ese_compression_size integer`.
4. To convert the trained AnglE models to `sentence-transformers`, please run `python scripts/convert_to_sentence_transformers.py --help` for more details.
