# [AnglE📐: Angle-optimized Text Embeddings](https://arxiv.org/abs/2309.12871)

> It is Angle 📐, not Angel 👼.

🔥 **A New SOTA** for Semantic Textual Similarity! 

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
<summary>📊 Click to show main results of AnglE</summary>
<p align='center'>
<img src='assets/angle-results.png'>
</p>
</details>

## 🤗 Pretrained Models

| 🤗 HF | Backbone | LLM | Language | Use Prompt | Avg Score. |
|----|------|------|------|------|------|
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) |  NousResearch/Llama-2-7b-hf | Y | EN | Y | **85.96** |
| [SeanLee97/angle-llama-7b-nli-20231027](https://huggingface.co/SeanLee97/angle-llama-7b-nli-20231027/tree/main) |  NousResearch/Llama-2-7b-hf | Y | EN | Y | 85.90 |



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

## Results

### English STS Results

| Model | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
| ------- |-------|-------|-------|-------|-------|--------------|-----------------|-------|
| [SeanLee97/angle-llama-7b-nli-20231027](https://huggingface.co/SeanLee97/angle-llama-7b-nli-20231027) | 78.68 | 90.58 | 85.49 | 89.56 | 86.91 |    88.92     |      81.18      | 85.90 |
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) | 79.00 | 90.56 | 85.79 | 89.43 | 87.00 |    88.97     |      80.94      | **85.96** |


## Usage

### Angle-LLaMA

1) AnglE

Install AnglE first

```bash
python pip install -U angle-emb
```

```python
from angle_emb import AnglE

angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2')
angle.set_prompt()
print('prompt:', angle.prompt)
vec = angle.encode({'text': 'hello world'}, to_numpy=True)
print(vec)
vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True)
print(vecs)
```

2) transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

peft_model_id = 'SeanLee97/angle-llama-7b-nli-v2'
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

## Train Custom AnglE Model

### 1. Train NLI

1) Prepare your gpu environment

2) Install python dependencies

```bash
python -m pip install -r requirements.txt
```

3) Download data

- Download multi_nli + snli:

```bash
$ cd data
$ sh download_data.sh
```
- Download sts datasets

```bash
$ cd SentEval/data/downstream
$ bash download_dataset.sh
```

### 2. Train w/ `train_angle.py`
The training interface is still messy, we are working on making it better. Currently you can modify `train_angle.py` to train your own models.

### 3. Custom Train

Coming soon!


# Citation

You are welcome to use our code and pre-trained models. If you use our code and pre-trained models, please support us by citing our work as follows:

```bibtex
@article{li2023angle,
  title={AnglE-Optimized Text Embeddings},
  author={Li, Xianming and Li, Jing},
  journal={arXiv preprint arXiv:2309.12871},
  year={2023}
}
```

When using our pre-trained LLM-based models and using `xxx in one word:` prompt, it is recommended to cite the following work in addition to the above citation:

```bibtex
@article{jiang2023scaling,
  title={Scaling Sentence Embeddings with Large Language Models},
  author={Jiang, Ting and Huang, Shaohan and Luan, Zhongzhi and Wang, Deqing and Zhuang, Fuzhen},
  journal={arXiv preprint arXiv:2307.16645},
  year={2023}
}
```
