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
CUDA_VISIBLE_DEVICES=0 python eval_nli.py \
--prompt 'Summarize sentence "{text}" in one word:"' \
--model_name_or_path NousResearch/Llama-2-7b-hf \
--lora_weight ckpts/NLI-STS-angle-llama-7b \
--apply_bfloat16
```