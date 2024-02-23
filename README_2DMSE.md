# 🪆 2D Matryoshka Sentence Embeddings

> Paper: https://arxiv.org/abs/2402.14776

# Usage

**⚠️ The Document is Working in Progress!**


Example:

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 angle-trainer \
--model_name_or_path WhereIsAI/UAE-Large-V1 \
--train_name_or_path data.jsonl --save_dir ckpts/custom-UAE-2dmse \
--w2 20.0 --w1 1. --w3 1. --angle_tau 20.0 --learning_rate 1e-5 --maxlen 128 \
--workers 16 \
--pooling_strategy all \
--epochs 1 \
--batch_size 16 \
--apply_tdmse 1 \
--fixed_teacher_name_or_path WhereIsAI/UAE-Large-V1 \
--logging_steps 1000 \
--warmup_steps 100 \
--is_llm 0 \
--save_steps 1000 --seed -1 --gradient_accumulation_steps 6 --fp16 1
```

The `--apply_tdmse 1` is required.


# Citation

```bibtex
@article{li20242d,
  title={2D Matryoshka Sentence Embeddings},
  author={Xianming Li and Zongxi Li and Jing Li and Haoran Xie and Qing Li},
  journal={arXiv preprint arXiv:2402.14776},
  year={2024}
}

@article{li2023angle,
  title={AnglE-optimized Text Embeddings},
  author={Li, Xianming and Li, Jing},
  journal={arXiv preprint arXiv:2309.12871},
  year={2023}
}
```