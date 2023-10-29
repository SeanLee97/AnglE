<small>[EN](README.md) | 简体中文 </small>

# [AnglE📐: Angle-optimized Text Embeddings](https://arxiv.org/abs/2309.12871)

> It is Angle 📐, not Angel 👼.

🔥 基于 AnglE 开箱即用的文本向量库，支持中英双语，可用于文本相似度计算、检索召回、匹配等场景。代码基于 🤗transformers 构建，提供易用的微调接口，可在 3090Ti、 4090 等消费级 GPU 上微调 LLaMA-7B 模型，支持多卡分布式训练。


<a href="https://arxiv.org/abs/2309.12871">
    <img src="https://img.shields.io/badge/Arxiv-2306.06843-yellow.svg?style=flat-square" alt="https://arxiv.org/abs/2309.12871" />
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


## 模型列表

| 🤗 HF | Backbone | LLM | Language | Prompt | Datasets | Pooling Strategy |
|----|------|------|------|------|------|------|
| [SeanLee97/angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) |  NousResearch/Llama-2-7b-hf | Y | EN | `Prompts.A` | multi_nli + snli | last token |
| [SeanLee97/angle-llama-7b-nli-20231027](https://huggingface.co/SeanLee97/angle-llama-7b-nli-20231027) |  NousResearch/Llama-2-7b-hf | Y | EN | `Prompts.A` | multi_nli + snli | last token |
| [SeanLee97/angle-bert-base-uncased-nli-en-v1](https://huggingface.co/SeanLee97/angle-bert-base-uncased-nli-en-v1) |  bert-base-uncased | N | EN | N | multi_nli + snli | `cls_avg` |
| [SeanLee97/angle-roberta-wwm-base-zhnli-v1](https://huggingface.co/SeanLee97/angle-roberta-wwm-base-zhnli-v1) |  hfl/chinese-roberta-wwm-ext | N | ZH-CN | N | zh_nli_all | `cls` |
| [SeanLee97/angle-llama-7b-zhnli-v1](https://huggingface.co/SeanLee97/angle-llama-7b-zhnli-v1) |  NousResearch/Llama-2-7b-hf | Y | ZH-CN | `Prompts.B` | zh_nli_all | last token |


## 模型效果

> 中文文档只列出中文模型的结果，英文请参照主文档

### 迁移 STS

使用 AnglE 在 [shibing624/nli-zh-all](https://huggingface.co/datasets/shibing624/nli-zh-all) 上训练 (加入了部分 [shibing624/nli_zh](https://huggingface.co/datasets/shibing624/nli_zh) train set 的数据)，并在以下 7 个数据集的 test set 上评测。


| 模型 | ATEC | BQ	| LCQMC | PAWSX | STS-B | SOHU-dd | SOHU-dc | Avg. |
| ------- |-------|-------|-------|-------|-------|--------------|-----------------|-------|
| ^[shibing624/text2vec-bge-large-chinese](https://huggingface.co/shibing624/text2vec-bge-large-chinese) | 38.41 | 61.34 | 71.72 | 35.15 | 76.44 | 71.81 | 63.15 | 59.72 |
| ^[shibing624/text2vec-base-chinese-paraphrase](https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase) |	44.89 | 63.58 | 74.24 | 40.90 | 78.93 | 76.70 | 63.30 | 63.08 |
| [SeanLee97/angle-roberta-wwm-base-zhnli-v1](https://huggingface.co/SeanLee97/angle-roberta-wwm-base-zhnli-v1) | 49.49 | 72.47 | 78.33 | 59.13 | 77.14 |    72.36     |      60.53      | **67.06** |
| [SeanLee97/angle-llama-7b-zhnli-v1](https://huggingface.co/SeanLee97/angle-llama-7b-zhnli-v1) | 50.44 | 71.95 | 78.90 | 56.57 | 81.11 | 68.11 | 52.02 | 65.59 |

^ 表示 baseline，结果来自：https://github.com/shibing624/text2vec

AnglE 的评测可参照 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OvvBu4c3pfkf_XcVuFyCqsAqCUArtZZ8#scrollTo=xFjso-JLNsKZ)

> 发现：LLaMA2-7B 的中文表达能力还有待提升。
 

### 非迁移 STS

在数据集各自的 train set 上训练，各自的 test set 上评测，效果如下表：

| 模型 | ATEC | BQ	| LCQMC | PAWSX | STS-B | Avg. |
| ------- |-------|-------|-------|-------|-------|--------------|
| ^CoSENT-bert-base | 49.74 | 72.38 | 78.69 | 60.00 | 79.27 | 68.01 |
| ^CoSENT-macbert-base | 50.39 | 72.93 | 79.17 | 60.86 | 79.30 | 68.53 |
| ^CoSENT-roberta-ext | 50.81 | 71.45 | 79.31 | 61.56 | 79.96 | 68.61 |
| AnglE-roberta-wwm-ext | **51.30** | **73.35** | **79.53** | **62.46** | **80.83** | **69.49** |

^ 为 baseline 模型，结果来自：[text2vec](https://github.com/shibing624/text2vec#%E4%B8%AD%E6%96%87%E5%8C%B9%E9%85%8D%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E8%AF%84%E6%B5%8B%E7%BB%93%E6%9E%9C)


AnglE-roberta-wwm-ext 各数据集的微调及评估代码如下：
- ATEC: [examples/Angle-ATEC.ipynb](examples/Angle-ATEC.ipynb)
- BQ: [examples/Angle-BQ.ipynb](examples/Angle-BQ.ipynb)
- LCQMC: [examples/Angle-LCQMC.ipynb](examples/Angle-LCQMC.ipynb)
- PAWSX: [examples/Angle-PAWSX.ipynb](examples/Angle-PAWSX.ipynb)
- SST-B: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzuaZjdkKqL_JasQnSGZ3g2H3H2aR6yG?usp=sharing)


## 用法

AnglE 支持两种调用方式，一种是 AnglE 提供的库，另一种是使用 huggingface transformers。推荐使用 AnglE 提供的库，可以省去较多调用步骤。要使用 AnglE 库需要先安装 `angle_emb`，如下：

```bash
python -m pip install -U angle-emb
```

### Angle-LLaMA

1) AnglE
```python
from angle_emb import AnglE, Prompts

angle = AnglE.from_pretrained('NousResearch/Llama-2-7b-hf', pretrained_lora_path='SeanLee97/angle-llama-7b-zhnli-v1')
# 请选择对应的 prompt，此模型对应 Prompts.B
print('All predefined prompts:', Prompts.list_prompts())
angle.set_prompt(prompt=Prompts.B)
print('prompt:', angle.prompt)
vec = angle.encode({'text': '你好世界'}, to_numpy=True)
print(vec)
vecs = angle.encode([{'text': '你好世界1'}, {'text': '你好世界2'}], to_numpy=True)
print(vecs)
```

2) transformers

```python
from angle_emb import Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

peft_model_id = 'SeanLee97/angle-llama-7b-zhnli-v1'
config = PeftConfig.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).bfloat16().cuda()
model = PeftModel.from_pretrained(model, peft_model_id).cuda()

def decorate_text(text: str):
    return Prompts.B.format(text=text)

inputs = '你好世界'
tok = tokenizer([decorate_text(inputs)], return_tensors='pt')
for k, v in tok.items():
    tok[k] = v.cuda()
vec = model(output_hidden_states=True, **tok).hidden_states[-1][:, -1].float().detach().cpu().numpy()
print(vec)
```

### Angle-BERT

1) AnglE
```python
from angle_emb import AnglE

angle = AnglE.from_pretrained('SeanLee97/angle-roberta-wwm-base-zhnli-v1', pooling_strategy='cls').cuda()
vec = angle.encode('你好世界', to_numpy=True)
print(vec)
vecs = angle.encode(['你好世界1', '你好世界2'], to_numpy=True)
print(vecs)
```

2) transformers

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = 'SeanLee97/angle-roberta-wwm-base-zhnli-v1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).cuda()

inputs = '你好世界'
tok = tokenizer([inputs], return_tensors='pt')
for k, v in tok.items():
    tok[k] = v.cuda()
hidden_state = model(**tok).last_hidden_state
vec = hidden_state[:, 0]
print(vec)
```

## 微调模型

只需要准备好数据即可快速微调。数据格式必须要转成 `datasets.Dataset` （使用方式请参照官方文档 [Datasets](https://huggingface.co/docs/datasets/index)）且必须要提供 `text1`, `text2`, `label` 三列。

下面以微调中文版本 STS-B 为例，演示整个微调训练及评估的过程。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzuaZjdkKqL_JasQnSGZ3g2H3H2aR6yG?usp=sharing)


```python
from datasets import load_dataset
from angle_emb import AnglE, AngleDataTokenizer

# 1. 加载模型
angle = AnglE.from_pretrained('hfl/chinese-roberta-wwm-ext', max_length=128, pooling_strategy='cls').cuda()

# 2. 加载数据并转换数据
ds = load_dataset('shibing624/nli_zh', 'STS-B')
ds = ds.rename_column('sentence1', 'text1')
ds = ds.rename_column('sentence2', 'text2')
ds = ds.select_columns(["text1", "text2", "label"])
train_ds = ds['train'].shuffle().map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
valid_ds = ds['validation'].map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)
test_ds = ds['test'].map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)

# 3. 训练
angle.fit(
    train_ds=train_ds,
    valid_ds=valid_ds,
    output_dir='ckpts/sts-b',
    batch_size=64,
    epochs=5,
    learning_rate=3e-5,
    save_steps=100,
    eval_steps=1000,
    warmup_steps=0,
    gradient_accumulation_steps=1,
    loss_kwargs={
        'w1': 1.0,
        'w2': 1.0,
        'w3': 1.0,
        'cosine_tau': 20,
        'ibn_tau': 20,
        'angle_tau': 1.0
    },
    fp16=True,
    logging_steps=100
)

# 4. 加载最优模型评估效果
angle = AnglE.from_pretrained('hfl/chinese-roberta-wwm-ext', pretrained_model_path='ckpts/sts-b/best-checkpoint').cuda()
corrcoef, accuracy = angle.evaluate(test_ds, device=angle.device)
print('corrcoef:', corrcoef)
```

# 引用 Citation

如果你有使用我们的代码及预训练模型，欢迎给我们三连，三连方式为：
1) 给本项目 GitHub 加个 star
2) 粘贴以下引用信息到你 paper 的 bibtex
3) 在你的 paper 正文中引用

```bibtex
@article{li2023angle,
  title={AnglE-Optimized Text Embeddings},
  author={Li, Xianming and Li, Jing},
  journal={arXiv preprint arXiv:2309.12871},
  year={2023}
}
```

当你使用我们的 LLM 模型且使用的 prompt 为 `xxx in one word:` 时，建议顺带引用以下文章:

```bibtex
@article{jiang2023scaling,
  title={Scaling Sentence Embeddings with Large Language Models},
  author={Jiang, Ting and Huang, Shaohan and Luan, Zhongzhi and Wang, Deqing and Zhuang, Fuzhen},
  journal={arXiv preprint arXiv:2307.16645},
  year={2023}
}
```


# FAQ

1) 为什么不使用国产 LLM？

各个都遥遥领先，选择困难症，标准太多，没时间搞，大家可以自行尝试。


2) 模型可商用嘛？

在没有特别声明下，可商用，但请遵循 backbone 模型本身的协议。


3) 代码跟我的环境冲突咋办？

本代码基于 transformers 构建，transformers 能跑通，本代码应该也能跑通，如果出问题请尝试按照 transformers 的方向解决。
时间因素，只能甩锅 transformers，请海涵。


4) 领域效果不佳咋办？

领域效果不佳请自己标数据微调解决，时间宝贵，我们不做免费外包。咨询收费，定制收费 （有偿咨询通道📮 xmlee97@gmail.com）。
