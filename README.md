<small>EN | [简体中文](README_zh.md) </small>

# AnglE 📐
> <small>Sponsored by <a href="https://www.mixedbread.ai/">Mixedbread</a></small>

**For more detailed usage, please read the 📘 document:** https://angle.readthedocs.io/en/latest/index.html

<a href="https://arxiv.org/abs/2309.12871">
    <img src="https://img.shields.io/badge/Arxiv-2309.12871-yellow.svg?style=flat-square" alt="https://arxiv.org/abs/2309.12871" />
</a>
<a href="https://pypi.org/project/angle_emb/">
    <img src="https://img.shields.io/pypi/v/angle_emb?style=flat-square" alt="PyPI version" />
</a>
<a href="https://pypi.org/project/angle_emb/">
    <img src="https://img.shields.io/pypi/dm/angle_emb?style=flat-square" alt="PyPI Downloads" />
</a>
<a href="https://angle.readthedocs.io/en/latest/index.html">
    <img src="https://readthedocs.org/projects/angle/badge/?version=latest&style=flat-square" alt="Read the docs" />
</a>


📢 **Train/Infer Powerful Sentence Embeddings with AnglE.**
This library is from the paper: [AnglE: Angle-optimized Text Embeddings](https://arxiv.org/abs/2309.12871). It allows for training state-of-the-art BERT/LLM-based sentence embeddings with just a few lines of code. AnglE is also a general sentence embedding inference framework, allowing for infering a variety of transformer-based sentence embeddings.

## ✨ Features

**Loss**:
- 📐 AnglE loss (ACL24)
- ⚖ Contrastive loss
- 📏 CoSENT loss
- ☕️ Espresso loss (ICLR 2025, a.k.a 2DMSE, detail: [README_ESE](README_ESE.md))

**Backbones**:
- BERT-based models (BERT, RoBERTa, ModernBERT, etc.)
- LLM-based models (LLaMA, Mistral, Qwen, etc.)
- Bi-directional LLM-based models (LLaMA, Mistral, Qwen, OpenELMo, etc.. refer to: https://github.com/WhereIsAI/BiLLM)

**Training**:
- Single-GPU training
- Multi-GPU training


> <a href="http://makeapullrequest.com"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="http://makeapullrequest.com" /></a> 
    More features will be added in the future. 

## 🏆 Achievements

📅  May 16, 2024 | Paper "[AnglE: Angle-optimized Text Embeddings](https://arxiv.org/abs/2309.12871)" is accepted by ACL 2024 Main Conference.

📅  Mar 13, 2024 | Paper "[BeLLM: Backward Dependency Enhanced Large Language Model for Sentence Embeddings](https://arxiv.org/abs/2311.05296)" is accepted by NAACL 2024 Main Conference.


📅  Mar 8, 2024 | 🍞 [mixedbread's embedding](https://www.mixedbread.ai/blog/mxbai-embed-large-v1) ([mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)) achieves SOTA on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) with an average score of **64.68**! The model is trained using AnglE. Congrats mixedbread!


📅  Dec 4, 2023 | Our universal sentence embedding [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) achieves SOTA on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) with an average score of **64.64**! The model is trained using AnglE.


📅 Dec, 2023 | AnglE achieves SOTA performance on the STS Bechmark Semantic Textual Similarity! 


## 🤗 Official Pretrained Models

BERT-based models:

|  🤗 HF | Max Tokens | Pooling Strategy | Scenario |
|----|------|------|------|
| [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1) | 512 | cls | English, General-purpose |
| [WhereIsAI/UAE-Code-Large-V1](https://huggingface.co/WhereIsAI/UAE-Code-Large-V1) |  512 | cls | Code Similarity |
| [WhereIsAI/pubmed-angle-base-en](https://huggingface.co/WhereIsAI/pubmed-angle-base-en) |  512 | cls | Medical Similarity |
| [WhereIsAI/pubmed-angle-large-en](https://huggingface.co/WhereIsAI/pubmed-angle-large-en) |  512 | cls | Medical Similarity |

LLM-based models:

| 🤗 HF (lora weight) | Backbone | Max Tokens | Prompts |  Pooling Strategy | Scenario  |
|----|------|------|------|------|------|
| [SeanLee97/angle-llama-13b-nli](https://huggingface.co/SeanLee97/angle-llama-13b-nli) | NousResearch/Llama-2-13b-hf | 4096 | `Prompts.A` | last token | English, Similarity Measurement | 
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) | NousResearch/Llama-2-7b-hf | 4096 | `Prompts.A` | last token | English, Similarity Measurement | 


**💡 You can find more third-party embeddings trained with AnglE in [HuggingFace Collection](https://huggingface.co/collections/SeanLee97/angle-based-embeddings-669a181354729d168a6ead9b)**


## 🚀 Quick Start

### ⬇️ Installation

```bash
pip install -U angle-emb
```

### ⌛ Infer BERT-based Model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)


1) **With Prompts**: You can specify a prompt with `prompt=YOUR_PROMPT` in `encode` method. The prompt should use `{text}` as the placeholder. We provide a set of predefined prompts in `Prompts` class, you can check them via `Prompts.list_prompts()`.

```python
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity


angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
# For retrieval tasks, we use `Prompts.C` as the prompt for the query when using UAE-Large-V1 (no need to specify prompt for documents).
qv = angle.encode(['what is the weather?'], to_numpy=True, prompt=Prompts.C)
doc_vecs = angle.encode([
    'The weather is great!',
    'it is rainy today.',
    'i am going to bed'
], to_numpy=True)

for dv in doc_vecs:
    print(cosine_similarity(qv[0], dv))
```

2) **Without Prompts**: no need to specify a prompt. Just input a list of strings or a single string.

```python
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
```


### ⌛ Infer LLM-based Models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)

If the pretrained weight is a LoRA-based model, you need to specify the backbone via `model_name_or_path` and specify the LoRA path via the `pretrained_lora_path` in `from_pretrained` method. **You must manually set `is_llm=True`** for LLM models.

```python
import torch
from angle_emb import AnglE, Prompts
from angle_emb.utils import cosine_similarity

angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                              pretrained_lora_path='SeanLee97/angle-llama-7b-nli-v2',
                              pooling_strategy='last',
                              is_llm=True,
                              torch_dtype=torch.float16).cuda()
print('All predefined prompts:', Prompts.list_prompts())
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
], prompt=Prompts.A)

for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```


### ⌛ Infer BiLLM-based Models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)

Specify `apply_billm` and `billm_model_class` to load and infer billm models


```python
import os
# set an environment variable for billm start index
os.environ['BiLLM_START_INDEX'] = '31'

import torch
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity

# specify `apply_billm` and `billm_model_class` to load billm models
# You must manually set is_llm=True for LLM models
angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf',
                              pretrained_lora_path='SeanLee97/bellm-llama-7b-nli',
                              pooling_strategy='last',
                              is_llm=True,
                              apply_billm=True,
                              billm_model_class='LlamaForCausalLM',
                              torch_dtype=torch.float16).cuda()

doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
], prompt='The representative word for sentence {text} is:"')

for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```


### ⌛ Infer Espresso/Matryoshka Models
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QJcA2Mvive4pBxWweTpZz9OgwvE42eJZ?usp=sharing)

Specify `layer_index` and `embedding_size` to truncate embeddings.


```python
from angle_emb import AnglE
from angle_emb.utils import cosine_similarity


angle = AnglE.from_pretrained('mixedbread-ai/mxbai-embed-2d-large-v1', pooling_strategy='cls').cuda()
# truncate layer
angle = angle.truncate_layer(layer_index=22)
# specify embedding size to truncate embeddings
doc_vecs = angle.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
], embedding_size=768)

for i, dv1 in enumerate(doc_vecs):
    for dv2 in doc_vecs[i+1:]:
        print(cosine_similarity(dv1, dv2))
```

### ⌛ Infer Third-party Models

You can load any transformer-based third-party models such as `mixedbread-ai/mxbai-embed-large-v1`, `sentence-transformers/all-MiniLM-L6-v2`, and `BAAI/bge-large-en-v1.5` using `angle_emb`.

Here is an example:

```python
from angle_emb import AnglE

model = AnglE.from_pretrained('mixedbread-ai/mxbai-embed-large-v1', pooling_strategy='cls').cuda()
vec = model.encode('hello world', to_numpy=True)
print(vec)
```

## Batch Inference

It is recommended to use Mixedbread's `batched` library to speed up the inference process.

```bash
python -m pip install batched
```

```python
import batched
from angle_emb import AnglE

model = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy='cls').cuda()
model.encode = batched.dynamically(model.encode, batch_size=64)

vecs = model.encode([
    'The weather is great!',
    'The weather is very good!',
    'i am going to bed'
] * 50)
```

## 🕸️ Custom Train

💡 For more details, please refer to the [training and fintuning](https://angle.readthedocs.io/en/latest/notes/training.html).


### 🗂️ 1. Data Preparation

We currently support three dataset formats:

1) **Format A** (Pair with Label): A pair format with three columns: `text1`, `text2`, and `label`. The `label` should be a similarity score (e.g., 0-1).

2) **Format B** (Query-Positive): A pair format with two columns: `query` and `positive`. Both `query` and `positive` can be either `str` or `List[str]` (if list, one will be randomly sampled during training).

3) **Format C** (Query-Positive-Negative): A triple format with three columns: `query`, `positive`, and `negative`. All three fields can be either `str` or `List[str]` (if list, one will be randomly sampled during training).

You need to prepare your data into huggingface `datasets.Dataset` in one of these formats.

### 🚂 2. Train with CLI [Recommended]

Use `angle-trainer` to train your AnglE model in cli mode. 

1) Single gpu training:

Usage: 

```bash
CUDA_VISIBLE_DEVICES=0 angle-trainer --help
```

2) Multi-gpu training:

using FSDP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
--multi_gpu \
--num_processes 4 \
--main_process_port 2345 \
--config_file examples/FSDP/fsdp_config.yaml \
-m angle_emb.angle_trainer \
--gradient_checkpointing 1 \
--use_reentrant 0 \
...
```

normal training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
--multi_gpu \
--num_processes 4 \
--main_process_port 2345 \
-m angle_emb.angle_trainer --help
```

More training examples can be found here [examples/Training](examples/Training)


### 🚂 3. Custom Train

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h28jHvv_x-0fZ0tItIMjf8rJGp3GcO5V?usp=sharing)


```python
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
```

### 💡 Others

- To enable `llm` training, you **must** manually specify `--is_llm 1` and configure appropriate LoRA hyperparameters.
- To enable `billm` training, please specify `--apply_billm 1` and configure appropriate `billm_model_class` such as `LLamaForCausalLM` (refer to: https://github.com/WhereIsAI/BiLLM?tab=readme-ov-file#usage).
- To enable espresso sentence embeddings (ESE), please specify `--apply_ese 1` and configure appropriate ESE hyperparameters via `--ese_kl_temperature float` and `--ese_compression_size integer`.
- To apply prompts during training:
  - Use `--text_prompt` for Format A (applies to both text1 and text2)
  - Use `--query_prompt` for query field in Format B/C
  - Use `--doc_prompt` for positive/negative fields in Format B/C
- To convert the trained AnglE models to `sentence-transformers`, please run `python scripts/convert_to_sentence_transformers.py --help` for more details.


## 💡 4. Fine-tuning Tips

For more details, please refer to the [documentation](https://angle.readthedocs.io/en/latest/notes/training.html#fine-tuning-tips).

1️⃣ If your dataset format is **Format A** (text1, text2, label), it is recommended to slightly increase the weight for `cosine_w` or slightly decrease the weight for `ibn_w`.

2️⃣ If your dataset format is **Format B** (query, positive), only `ibn_w` and `ibn_tau` are effective. You don't need to tune other parameters.

3️⃣ If your dataset format is **Format C** (query, positive, negative), it is recommended to set `cosine_w` to 0, and set `angle_w` to a small value like 0.02. Be sure to set `cln_w` and `ibn_w`.

4️⃣ To alleviate information forgetting in fine-tuning, it is better to specify the `teacher_name_or_path`. If the `teacher_name_or_path` equals `model_name_or_path`, it will conduct self-distillation. **It is worth to note that** `teacher_name_or_path` has to have the same tokenizer as `model_name_or_path`. Or it will lead to unexpected results.


## 5. Finetuning and Infering AnglE with `sentence-transformers`

- **Training:** SentenceTransformers also provides a implementation of [AnglE loss](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#angleloss). **But it is partially implemented and may not work well as the official code. We recommend to use the official `angle_emb` for fine-tuning AnglE model.**

- **Infering:** If your model is trained with `angle_emb`, and you want to use it with `sentence-transformers`. You can convert it to `sentence-transformers` model using the script `examples/convert_to_sentence_transformers.py`.


# 🫡 Citation

If you use our code and pre-trained models, please support us by citing our work as follows:

```bibtex
@article{li2023angle,
  title={AnglE-optimized Text Embeddings},
  author={Li, Xianming and Li, Jing},
  journal={arXiv preprint arXiv:2309.12871},
  year={2023}
}
```

# 📜 ChangeLogs

| 📅 | Description |
|----|------|
| 2025 Jan |  **Major refactoring**: Removed `AngleDataTokenizer`, simplified data pipeline, removed auto-detection of LLM models, added separate prompt parameters |
| 2024 May 21 |  support Espresso Sentence Embeddings  |
| 2024 Feb 7 |  support training with only positive pairs (Format C: query, positive)  |
| 2023 Dec 4 |  Release a universal English sentence embedding model: [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)  |
| 2023 Nov 2 |  Release an English pretrained model: `SeanLee97/angle-llama-13b-nli` |
| 2023 Oct 28 |  Release two chinese pretrained models: `SeanLee97/angle-roberta-wwm-base-zhnli-v1` and `SeanLee97/angle-llama-7b-zhnli-v1`; Add chinese README.md |

# 📧 Contact

If you have any questions or suggestions, please feel free to contact us via email: xmlee97@gmail.com

# © License

This project is licensed under the MIT License.
For the pretrained models, please refer to the corresponding license of the models.
