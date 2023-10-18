# -*- coding: utf-8 -*-

import sys

from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

if len(sys.argv) != 3:
    print('python upload_model.py local_path remote_path')
    sys.exit()


local_peft_model_id = sys.argv[1]
remote_peft_model_id = sys.argv[2]

print('local peft model path:', local_peft_model_id)
print('remote peft model path:', remote_peft_model_id)

print('loading model...')
config = PeftConfig.from_pretrained(local_peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, local_peft_model_id)
print('start to push...')
model.push_to_hub(remote_peft_model_id, use_auth_token=True)
