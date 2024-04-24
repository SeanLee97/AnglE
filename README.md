<small>EN | [简体中文](README_zh.md) </small>

# [AnglE📐: Angle-optimized Text Embeddings](https://arxiv.org/abs/2309.12871)

> It is Angle 📐, not Angel 👼.

📢 **Train/Infer Powerful Sentence Embedding Models with AnglE.**
AnglE enables you to train state-of-the-art BERT-based or LLM-based sentence embeddings with just a few lines of code. 
AnglE is also a general inference framework for sentence embedding, allowing you to infer a variety of transformer-based sentence embeddings.


## 🏆 Achievements

<a href="https://arxiv.org/abs/2309.12871">
    <img src="https://img.shields.io/badge/Arxiv-2309.12871-yellow.svg?style=flat-square" alt="https://arxiv.org/abs/2309.12871" />
</a>
<a href="https://pypi.org/project/angle_emb/">
    <img src="https://img.shields.io/pypi/v/angle_emb?style=flat-square" alt="PyPI version" />
</a>
<a href="https://pypi.org/project/angle_emb/">
    <img src="https://img.shields.io/pypi/dm/angle_emb?style=flat-square" alt="PyPI Downloads" />
</a>
<a href="http://makeapullrequest.com">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="http://makeapullrequest.com" />
</a>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sick-r-1)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sick-r-1?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts16)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts16?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts15)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts15?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts14)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts14?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts13)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts13?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts12)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts12?p=angle-optimized-text-embeddings)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/angle-optimized-text-embeddings/semantic-textual-similarity-on-sts-benchmark)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark?p=angle-optimized-text-embeddings)


📅  Mar 13, 2024 | Paper "[BeLLM: Backward Dependency Enhanced Large Language Model for Sentence Embeddings](https://arxiv.org/abs/2311.05296)" accepted by NAACL 2024 Main Conference.


📅  Mar 8, 2024 | 🍞 [mixedbread's embedding](https://www.mixedbread.ai/blog/mxbai-embed-large-v1) ([mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)) achieves SOTA on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) with an average score of **64.68**! The model is trained using AnglE.


📅  Dec 4, 2023 | Our universal sentence embedding [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) achieves SOTA on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) with an average score of **64.64**! The model is trained using AnglE.


📅 Dec, 2023 | **A New SOTA** for Semantic Textual Similarity! 


## 🤗 Official Pretrained Models
| 🤗 HF | LoRA Weight | Dependent Backbone | LLM | Language | Prompt | Pooling Strategy | Examples |
|----|------|------|------|------|------|------|------|
| [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) |  N | N | N | EN | `Prompts.C` for retrieval purposes, `None` for others | cls | [![Seach Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WOYD6f8gb_wpkUm_57K8pEDgjlGJd6oB?usp=drive_link) |
| [SeanLee97/angle-llama-13b-nli](https://huggingface.co/SeanLee97/angle-llama-13b-nli) | Y |  NousResearch/Llama-2-13b-hf | Y | EN | `Prompts.A` | last token | / | 
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) | Y |  NousResearch/Llama-2-7b-hf | Y | EN | `Prompts.A` | last token | / |
| [SeanLee97/angle-bert-base-uncased-nli-en-v1](https://huggingface.co/SeanLee97/angle-bert-base-uncased-nli-en-v1) | N |  N | N | EN | N | `cls_avg` | / |
 
<details><summary>💡 Tips</summary>
💡 If the selected model is a LoRA weight, it must specify the corresponding dependent backbone.

For our STS Experiment, please refer to https://github.com/SeanLee97/AnglE/tree/main/examples/NLI

</details>

## Results

### English STS Results

| Model | STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
| ------- |-------|-------|-------|-------|-------|--------------|-----------------|-------|
| [SeanLee97/angle-llama-7b-nli-20231027](https://huggingface.co/SeanLee97/angle-llama-7b-nli-20231027) | 78.68 | 90.58 | 85.49 | 89.56 | 86.91 |    88.92     |      81.18      | 85.90 |
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) | 79.00 | 90.56 | 85.79 | 89.43 | 87.00 |    88.97     |      80.94      | 85.96 |
| [SeanLee97/angle-llama-13b-nli](https://huggingface.co/SeanLee97/angle-llama-13b-nli)  | 79.33 | 90.65 | 86.89 | 90.45 | 87.32 |    89.69     |      81.32       | **86.52** |
| [SeanLee97/angle-bert-base-uncased-nli-en-v1](https://huggingface.co/SeanLee97/angle-bert-base-uncased-nli-en-v1) | 75.09 | 85.56 | 80.66 | 86.44 | 82.47 | 85.16 | 81.23 | 82.37 |


## 🚀 Quick Start

AnglE supports two APIs, one is the `transformers` API, the other is the `AnglE` API. If you want to use the `AnglE` API, please install AnglE first:

```bash
python -m pip install -U angle-emb
```

### 1. Load BERT-based Models

1) For Retrieval Purposes

For retrieval purposes, please use the prompt `Prompts.C` for query (not document).

```python
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity


angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
# when specify prompt, the inputs should be a list of dict with key 'text'
qv = angle.encode({'text': 'what is the weather?'}, to_numpy=True, prompt=Prompts.C)
doc_vecs = angle.encode([
    'The weather is great!',
    'it is rainy today.',
    'i am going to bed'
], to_numpy=True)

for dv in doc_vecs:
    print(cosine_similarity(qv[0], dv))
```

2) For non-Retrieval Purposes

```python
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity


angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
])

for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```

<details>
<summary>Difference between retrieval and non-retrieval sentence embeddings. [click to expand]</summary>

In UAE, we use different approaches for retrieval and non-retrieval tasks, each serving a different purpose. 

**Retrieval tasks aim to find relevant documents, and as a result, the related documents may not have strict semantic similarities to each other.**

For instance, when querying "How about ChatGPT?", the related documents are those that contain information related to "ChatGPT," such as "ChatGPT is amazing..." or "ChatGPT is bad....".

Conversely, **non-retrieval tasks, such as semantic textual similarity, require sentences that are semantically similar.**

For example, a sentence semantically similar to "How about ChatGPT?" could be "What is your opinion about ChatGPT?".

To distinguish between these two types of tasks, we use different prompts. 

For retrieval tasks, we use the prompt "Represent this sentence for searching relevant passages: {text}" (Prompts.C in angle_emb) for the query (**no need to apply it for the documents**). 

For non-retrieval tasks, we set the prompt to empty, i.e., just input your text without specifying a prompt.

So, if your scenario is retrieval-related, it is highly recommended to set the prompt with angle.set_prompt(prompt=Prompts.C). If not, leave the prompt empty or use angle.set_prompt(prompt=None).
</details>

### 2. Load LoRA-based Models

1) AnglE
```python
from angle_emb import AnglE, Prompts

angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                              pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
                              pooling_strategy='last')

print('All predefined prompts:', Prompts.list_prompts())
vec = angle.encode({'text': 'hello world'}, to_numpy=True, prompt=Prompts.A)
print(vec)
vecs = angle.encode([{'text': 'hello world1'}, {'text': 'hello world2'}], to_numpy=True, prompt=Prompts.A)
print(vecs)
```

### 3. Load Third-party Models w/ angle_emb

You can load any transformer-based third-party models such as `mixedbread-ai/mxbai-embed-large-v1`, `sentence-transformers/all-MiniLM-L6-v2`, and `BAAI/bge-large-en-v1.5` using `angle_emb`.

Here is an example:

```python
from angle_emb import AnglE

model = AnglE.from_pretrained('mixedbread-ai/mxbai-embed-large-v1', pooling_strategy='cls').cuda()
vec = model.encode('hello world', to_numpy=True)
print(vec)
```


## Custom Train

### 1. Data Prepation

We support two dataset formats:

1) `DatasetFormats.A`: it is a pair format with three columns: `text1`, `text2`, and `label` (0/1).

2) `DatasetFormats.B`: it is a triple format with three columns: `text`, `positive`, and `negative`. `positive` and `negative` store the positive and negative samples of `text`.

3) `DatasetFormats.C`: it is a pair format with two columns: `text`, `positive`. `positive` store the positive sample of `text`.

You need to prepare your data into huggingface `datasets.Dataset` in one of the formats in terms of your supervised data.

### 2. Train

Use `angle-trainer` to train your AnglE model in cli mode. Usage: `CUDA_VISIBLE_DEVICES=0 angle-trainer --help`


### 3. Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h28jHvv_x-0fZ0tItIMjf8rJGp3GcO5V?usp=sharing)


```python
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
        'angle_tau': 20.0
    },
    fp16=True,
    logging_steps=100
)

# 5. evaluate
corrcoef, accuracy = angle.evaluate(test_ds, device=angle.device)
print('corrcoef:', corrcoef)
```

### 4. Fine-tuning Tips 💡

1) if your dataset format is `DatasetFormats.A`, it is recommended to slightly increase the weight for `cosine_w` or slightly decrease the weight for `ibn_w`.

2) if your dataset format is `DatasetFormats.B`, it is recommended to set `cosine_w` to 0, and increase the weight for `ibn_w` such as 10 and 20. The `angle_tau` can be set to 20.0.

3) if your dataset format is `DatasetFormats.C`, only `ibn_w` and `ibn_tau` are effective. You don't need to tune other parameters.

4) To alleviate information forgetting in fine-tuning, it is better to specify the `teacher_name_or_path`. If the `teacher_name_or_path` equals `model_name_or_path`, it will conduct self-distillation. **It is worth to note that** `teacher_name_or_path` has to have the same tokenizer as `model_name_or_path`. Or it will lead to unexpected results.


# Citation

You are welcome to use our code and pre-trained models. If you use our code and pre-trained models, please support us by citing our work as follows:

```bibtex
@article{li2023angle,
  title={AnglE-optimized Text Embeddings},
  author={Li, Xianming and Li, Jing},
  journal={arXiv preprint arXiv:2309.12871},
  year={2023}
}
```

# ChangeLogs

| 📅 | Description |
|----|------|
| 2024 Feb 7 |  support training with only positive pairs (`DatasetFormats.C`)  |
| 2024 Jan 11 |  refactor to support `angle-trainer` and BeLLM  |
| 2023 Dec 4 |  Release a universal English sentence embedding model: [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)  |
| 2023 Nov 2 |  Release an English pretrained model: `SeanLee97/angle-llama-13b-nli` |
| 2023 Oct 28 |  Release two chinese pretrained models: `SeanLee97/angle-roberta-wwm-base-zhnli-v1` and `SeanLee97/angle-llama-7b-zhnli-v1`; Add chinese README.md |
