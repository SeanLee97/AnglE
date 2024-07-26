# -*- coding: utf-8 -*-


from angle_emb import AnglE
from datasets import load_dataset
from multiprocess import set_start_method


# configuration
n_gpus = 4
workers = 8
batch_size = 16

# init angle
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

# load dataset
ds = load_dataset('mteb/stsbenchmark-sts', split='test')
ds = ds.select_columns(('sentence1', 'sentence2'))


def encode(examples, rank):
    device = f"cuda:{rank}"
    angle.to(device)

    docs = [f'{s1} {s2}' for s1, s2 in zip(examples['sentence1'], examples['sentence2'])]
    examples['emb'] = angle.encode(docs).tolist()
    return examples


if __name__ == '__main__':

    # it is required to put the inference code in the main function.

    set_start_method('spawn')

    # map and encode
    ds = ds.map(encode, with_rank=True, num_proc=n_gpus, batched=True, batch_size=batch_size)

    print(ds)
    # ds.push_to_hub(xxx)
