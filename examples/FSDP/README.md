```bash
CUDA_VISIBLE_DEVICES=2,3 WANDB_MODE=disabled accelerate launch \
--multi_gpu \
--num_processes=2 \
--main_process_port 2345 \
--config_file fsdp_config.yaml \
-m angle_emb.angle_trainer \
--gradient_checkpointing 1 \
--use_reentrant 0 \
...
```

please ensure the following setting

```bash
--gradient_checkpointing 1 \
--use_reentrant 0 \
```
