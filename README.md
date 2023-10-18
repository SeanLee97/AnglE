# [AnglE📐: Angle-Optimized Text Embeddings](https://arxiv.org/abs/2309.12871)

> It is Angle 📐, not Angel 👼.

<h2 align='center'>🔥 A New SOTA Model for Semantic Textual Similarity! 
</h2>

<a href="https://arxiv.org/abs/2309.12871">
    <img src="https://img.shields.io/badge/Arxiv-2306.06843-yellow.svg?style=flat-square" alt="https://arxiv.org/abs/2309.12871" />
</a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sick-r-1)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sick-r-1?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts16)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts16?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts15)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts15?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts14)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts14?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts13)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts13?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts12)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts12?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts-benchmark)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark?p=angle-optimized-text-embeddings)


<details>
<summary align='center'>📊 Click to show main results of AnglE</summary>
<p align='center'>
<img src='assets/angle-results.png'>
</p>
</details>

## 🤗 Pretrained Models

| HF | Avg. |
|----|------|
| [SeanLee97/angle-llama-7b-nli-20231027](https://huggingface.co/SeanLee97/angle-llama-7b-nli-20231027/tree/main) |   0.8590   |


> <small>💬 The model above was trained using BERT's hyperparameters. Currently, We are working on searching for even better hyperparameters for Angle-LLaMA. We plan to release more advanced pre-trained models that will further enhance performance. Stay tuned ;)😉 </small>


**📝 Training Details:**

<details>
<summary>1) SeanLee97/angle-llama-7b-nli-20231027</summary>

We fine-tuned AnglE-LLaMA using 4 RTX 3090 Ti (24GB), the training script is as follows:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 train_angle.py \
--task NLI-STS --save_dir ckpts/NLI-STS-angle-llama-7b \
--w2 35 --learning_rate 2e-4 --maxlen 45 \
--lora_r 32 --lora_alpha 32 --lora_dropout 0.1 \
--save_steps 200 --batch_size 160 --seed 42 --do_eval 0 --load_kbit 4 --gradient_accumulation_steps 4 --epochs 1 
```

The evaluation script is as follows:

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval.py \
    --load_kbit 16 \
    --model_name_or_path NousResearch/Llama-2-7b-hf \
    --lora_weight SeanLee97/angle-llama-7b-nli-20231027
```

</details>


## Usage

### Angle-LLaMA

1) using transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

peft_model_id = 'SeanLee97/angle-llama-7b-nli-20231027'
config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).bfloat16().cuda()
model = PeftModel.from_pretrained(model, peft_model_id).cuda()

def decorate_text(text: str):
    return f'Summarize sentence "{text}" in one word:"'

inputs = 'hello world!'
tok = tokenizer([decorate_text(inputs)], return_tensors='pt')
for k, v in tok.items():
    tok[k] = v.cuda()
vec = model(output_hidden_states=True, **tok).hidden_states[-1][:, -1].float().detach().cpu().numpy()
print(vec)
```

2) using AnglE

Coming soon!


### Angle-BERTs

Coming soon!


## Train Custom AnglE Model

The training interface is still messy, we are working on making it better. Currently you can modify `train_angle.py` to train your own models.
