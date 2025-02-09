# Espresso Sentence Embeddings (previously known as 2DMSE)

> Paper: https://arxiv.org/abs/2402.14776 (ICLR 2025)


## Abstract

High-quality sentence embeddings are fundamental in many natural language processing (NLP) tasks, such as semantic textual similarity (STS) and retrieval-augmented generation (RAG). 
Nevertheless, most existing methods leverage fixed-length embeddings from full-layer language models, which lack the scalability to accommodate the diverse available resources across various applications.
Viewing this gap, we propose a novel sentence embedding model $\mathrm{Espresso}$ $\mathrm{Sentence}$ $\mathrm{Embeddings}$ (ESE) with two learning processes. 
First, the **learn-to-express** process encodes more salient representations to lower layers.
Second, the **learn-to-compress** process compacts essential features into the initial dimensions using Principal Component Analysis (PCA).
This way, ESE can scale model depth via the former process and embedding size via the latter.
Extensive experiments on STS and RAG suggest that ESE can effectively produce high-quality embeddings with less model depth and embedding size, enhancing embedding inference efficiency.

## Usage

### 1. Train with angle-emb
To enable espresso sentence embeddings (ESE), please specify `--apply_ese 1` and configure appropriate ESE hyperparameters via `--ese_kl_temperature float` and `--ese_compression_size integer`.

Here is an training example:

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=1234 -m angle_emb.angle_trainer \
--model_name_or_path WhereIsAI/UAE-Large-V1 \
--train_name_or_path SeanLee97/nli_for_simcse --save_dir ckpts/UAE-Large-Espresso \
--ibn_w 10.0 --cosine_w 0. --angle_w 1.0 --angle_tau 20.0 --learning_rate 1e-6 --maxlen 75 \
--workers 16 \
--pooling_strategy cls \
--epochs 1 \
--batch_size 128 \
--logging_steps 100 \
--warmup_steps 200 \
--save_steps 1000 \
--fp16 1 \
--gradient_accumulation_steps 4 \
--apply_ese 1 \
--ese_compression_size 128 \
--ese_kl_temperature 1.0
```

### 2. Train with sentence-transformers

Espresso has already been integrated into `sentence-transformers`; please refer to https://sbert.net/examples/training/adaptive_layer/README.html for training.


# Citation

```bibtex
article{li2024ese,
  title={Ese: Espresso sentence embeddings},
  author={Li, Xianming and Li, Zongxi and Li, Jing and Xie, Haoran and Li, Qing},
  journal={Preprint},
  year={2024}
}
```
