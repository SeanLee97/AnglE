# 🤗 HF Pretrained Models

[AnglE NLI Sentence Embedding](https://huggingface.co/collections/SeanLee97/angle-nli-sentence-embeddings-6646de386099d0472c5e21c0)

# English STS Results

| Model | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
| ------- |-------|-------|-------|-------|-------|--------------|-----------------|-------|
| [SeanLee97/angle-llama-7b-nli-20231027](https://huggingface.co/SeanLee97/angle-llama-7b-nli-20231027) | 78.68 | 90.58 | 85.49 | 89.56 | 86.91 |    88.92     |      81.18      | 85.90 |
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) | 79.00 | 90.56 | 85.79 | 89.43 | 87.00 |    88.97     |      80.94      | 85.96 |
| [SeanLee97/angle-llama-13b-nli](https://huggingface.co/SeanLee97/angle-llama-13b-nli)  | 79.33 | 90.65 | 86.89 | 90.45 | 87.32 |    89.69     |      81.32       | **86.52** |
| [SeanLee97/angle-bert-base-uncased-nli-en-v1](https://huggingface.co/SeanLee97/angle-bert-base-uncased-nli-en-v1) | 75.09 | 85.56 | 80.66 | 86.44 | 82.47 | 85.16 | 81.23 | 82.37 |


# Train NLI for STS Benchmark

## 1. Prepare your gpu environment

## 2. Install angle_emb

```bash
python -m pip install -U angle_emb
```

## 3. Download data

1) Download multi_nli + snli:

```bash
$ cd data
$ sh download_data.sh
```

2) Download STS datasets

```bash
$ cd SentEval/data/downstream
$ bash download_dataset.sh
```

## 4. Train script

1) use `train_angle.py`

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=1234 train_angle.py \
--task NLI-STS --save_dir ckpts/NLI-STS-angle-llama-7b \
--model_name NousResearch/Llama-2-7b-hf \
--w2 35 --learning_rate 1e-4 --maxlen 50 \
--lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
--save_steps 500 --batch_size 120 --seed 42 --do_eval 0 --load_kbit 4 --gradient_accumulation_steps 4 --epochs 1
```

2) use `angle-trainer`

You need to transform the  AllNLI dataset into jsonl format like {"text1": "", "text2": "", "label": 0/1}.
For the label, we set `entailment` to `1`, `contradiction` to `0`, and skip `neutral`.
Supposed the filename is `train.jsonl`, then you can train as follows:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=1234 angle-trainer \
--model_name_or_path NousResearch/Llama-2-7b-hf \
--train_name_or_path train.jsonl \
--save_dir ckpts/NLI-STS-angle-llama-7b \
--prompt_template 'Summarize sentence "{text}" in one word:"' \
--w2 35 --learning_rate 1e-4 --maxlen 50 \
--lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
--save_steps 500 --batch_size 120 --seed 42 --do_eval 0 --load_kbit 4 --gradient_accumulation_steps 4 --epochs 1
```

## 5. Evaluate STS Performance using SentEval

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval_nli.py \
--model_name_or_path NousResearch/Llama-2-7b-hf \
--lora_name_or_path SeanLee97/angle-llama-7b-nli \
--pooling_strategy last \
--is_llm 1
```
