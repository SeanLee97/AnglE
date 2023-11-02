# -*- coding: utf-8 -*-

import sys

from transformers import AutoModel, AutoTokenizer

if len(sys.argv) != 4:
    print('python upload_model.py backbone local_path remote_path')
    sys.exit()

backbone = sys.argv[1]
local_model_id = sys.argv[2]
remote_model_id = sys.argv[3]

print('local model path:', local_model_id)
print('remote model path:', remote_model_id)

print('loading model...')
model = AutoModel.from_pretrained(local_model_id)
tokenizer = AutoTokenizer.from_pretrained(backbone)
print('start to push...')
model.push_to_hub(remote_model_id, use_auth_token=True)
tokenizer.push_to_hub(remote_model_id, use_auth_token=True)
