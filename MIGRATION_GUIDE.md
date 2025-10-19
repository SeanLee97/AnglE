# Migration Guide: Upgrading to v0.6.0

This guide will help you upgrade your code to the latest version of AnglE, which includes significant refactoring to simplify the API and improve usability.

## 📋 Summary of Changes

### Breaking Changes

1. **Removed `AngleDataTokenizer`** - No need to pre-tokenize datasets
2. **Removed `DatasetFormats` class** - Use string literals instead
3. **Removed `check_llm()` function** - Set `is_llm` manually
4. **Removed command-line arguments**: `--coword_random_mask_rate`, `--fix_data`
5. **Renamed `--prompt_template`** to `--text_prompt` (Format A only)
6. **New prompt parameters**: `--query_prompt`, `--doc_prompt`
7. **Updated data format requirements**
8. **Training examples now use `accelerate launch`** instead of `torchrun`

---

## 🔄 Migration Steps

### 1. Update Your Data Pipeline

**Before (v0.5.x):**
```python
from datasets import load_dataset
from angle_emb import AnglE, AngleDataTokenizer

angle = AnglE.from_pretrained('model-name', max_length=128, pooling_strategy='cls').cuda()

ds = load_dataset('your-dataset')
# Had to pre-tokenize the dataset
train_ds = ds['train'].shuffle().map(
    AngleDataTokenizer(angle.tokenizer, angle.max_length), 
    num_proc=8
)

angle.fit(train_ds=train_ds, ...)
```

**After (v0.6.0):**
```python
from datasets import load_dataset
from angle_emb import AnglE

angle = AnglE.from_pretrained('model-name', max_length=128, pooling_strategy='cls').cuda()

ds = load_dataset('your-dataset')
# Just load the raw dataset - tokenization happens automatically!
train_ds = ds['train'].shuffle()

angle.fit(train_ds=train_ds, ...)
```

### 2. Update Dataset Format References

**Before (v0.5.x):**
```python
from angle_emb import DatasetFormats

# Format A
dataset_format = DatasetFormats.A

# Format B
dataset_format = DatasetFormats.B

# Format C
dataset_format = DatasetFormats.C
```

**After (v0.6.0):**
```python
# Use string literals directly
dataset_format = 'A'  # or 'B' or 'C'
```

### 3. Update Data Format Requirements

**Format A** (text1, text2, label):
- `text1`: str
- `text2`: str
- `label`: float (similarity score, e.g., 0-1)

**Format B** (query, positive):
```python
# Old field names (deprecated)
{"text": "...", "positive": "..."}

# New field names (required)
{"query": "...", "positive": "..."}

# Can also use lists - one will be randomly sampled
{"query": ["query1", "query2"], "positive": ["pos1", "pos2"]}
```

**Format C** (query, positive, negative):
```python
# Old field names (deprecated)
{"text": "...", "positive": "...", "negative": "..."}

# New field names (required)
{"query": "...", "positive": "...", "negative": "..."}

# Can also use lists - one will be randomly sampled
{"query": "...", "positive": ["pos1", "pos2"], "negative": ["neg1", "neg2"]}
```

**💡 Pro Tip: Use `--column_rename_mapping` to migrate old datasets without modification!**

If your existing dataset uses old field names, you can remap them using the `--column_rename_mapping` argument:

```bash
# Example: Remap "text" to "query"
angle-trainer \
  --train_name_or_path your-dataset \
  --column_rename_mapping "text:query" \
  ...

# You can also use it in Python code:
from angle_emb import AnglE

angle = AnglE.from_pretrained('model-name').cuda()

# Your old dataset with "text" field
ds = load_dataset('your-old-dataset')  # Has "text" field

angle.fit(
    train_ds=ds['train'],
    column_rename_mapping={"text": "query"},  # Remap on the fly!
    ...
)
```

### 4. Update LLM Model Initialization

**Before (v0.5.x):**
```python
# is_llm was auto-detected based on model name patterns
angle = AnglE.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    pretrained_lora_path='path/to/lora',
    pooling_strategy='last'
).cuda()
```

**After (v0.6.0):**
```python
# Must manually set is_llm=True
angle = AnglE.from_pretrained(
    'NousResearch/Llama-2-7b-hf',
    pretrained_lora_path='path/to/lora',
    pooling_strategy='last',
    is_llm=True  # ⚠️ Required!
).cuda()
```

### 5. Update Prompt Configuration

**Before (v0.5.x):**
```python
# Single prompt parameter for all texts
angle.fit(
    train_ds=train_ds,
    prompt_template="Represent this sentence: {text}",
    ...
)
```

**After (v0.6.0):**
```python
# Different prompts for different fields

# For Format A (text1, text2, label)
angle.fit(
    train_ds=train_ds,
    text_prompt="Represent this sentence: {text}",  # Applied to both text1 and text2
    ...
)

# For Format B/C (query, positive, negative)
angle.fit(
    train_ds=train_ds,
    query_prompt="query: {text}",      # Applied to query field
    doc_prompt="doc: {text}",          # Applied to positive/negative fields
    ...
)
```

### 6. Update Command-Line Training Scripts

**Before (v0.5.x):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nproc_per_node=4 \
--master_port=1234 \
-m angle_emb.angle_trainer \
--train_name_or_path dataset-name \
--model_name_or_path model-name \
--prompt_template "Represent: {text}" \
...
```

**After (v0.6.0):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=disabled accelerate launch \
--multi_gpu \
--num_processes 4 \
--main_process_port 1234 \
-m angle_emb.angle_trainer \
--train_name_or_path dataset-name \
--model_name_or_path model-name \
--query_prompt "query: {text}" \
--doc_prompt "doc: {text}" \
--is_llm 1 \  # If using LLM
...
```

### 7. Update Evaluation Code

**Before (v0.5.x):**
```python
# evaluate() returned tuple (corrcoef, accuracy)
corrcoef, accuracy = angle.evaluate(test_ds, device=angle.device)
print(f'corrcoef: {corrcoef}, accuracy: {accuracy}')
```

**After (v0.6.0):**
```python
# evaluate() returns only corrcoef for Format A datasets
corrcoef = angle.evaluate(test_ds)
print(f'Spearman corrcoef: {corrcoef}')
```

---

## 📝 Command-Line Argument Changes

### Removed Arguments
- `--coword_random_mask_rate` ❌
- `--fix_data` ❌
- `--prompt_template` ❌ (renamed)

### New/Renamed Arguments
- `--text_prompt` ✅ (replaces `--prompt_template`, Format A only)
- `--query_prompt` ✅ (for query field in Format B/C)
- `--doc_prompt` ✅ (for positive/negative fields in Format B/C)
- `--column_rename_mapping` ✅ (remap old column names to new ones)

### Important Notes
- `--is_llm 1` must be **explicitly set** for LLM models
- Use `accelerate launch` instead of `torchrun` for distributed training
- Use `--column_rename_mapping` to adapt old datasets without modifying them

---

## 🎯 Common Migration Scenarios

### Scenario 1: Training BERT Model on STS Data (Format A)

**Before:**
```python
from datasets import load_dataset
from angle_emb import AnglE, AngleDataTokenizer

angle = AnglE.from_pretrained('bert-base-uncased', max_length=128, pooling_strategy='cls').cuda()

ds = load_dataset('mteb/stsbenchmark-sts')
ds = ds.map(lambda obj: {"text1": obj["sentence1"], "text2": obj["sentence2"], "label": obj['score']})

train_ds = ds['train'].map(AngleDataTokenizer(angle.tokenizer, angle.max_length), num_proc=8)

angle.fit(train_ds=train_ds, ...)
```

**After:**
```python
from datasets import load_dataset
from angle_emb import AnglE

angle = AnglE.from_pretrained('bert-base-uncased', max_length=128, pooling_strategy='cls').cuda()

ds = load_dataset('mteb/stsbenchmark-sts')
ds = ds.map(lambda obj: {"text1": obj["sentence1"], "text2": obj["sentence2"], "label": obj['score']})

# No need to tokenize!
angle.fit(train_ds=ds['train'], ...)
```

### Scenario 2: Training with Query-Positive Pairs (Format B)

**Before:**
```python
# Dataset with "text" and "positive" fields
ds = [{"text": "query", "positive": "pos"}]
train_ds = Dataset.from_list(ds).map(AngleDataTokenizer(tokenizer, max_length), num_proc=8)
```

**After (Option 1 - Modify dataset):**
```python
# Must rename "text" to "query"
ds = [{"query": "query", "positive": "pos"}]
train_ds = Dataset.from_list(ds)  # No tokenization needed!

# Use query_prompt and doc_prompt
angle.fit(
    train_ds=train_ds,
    query_prompt="query: {text}",
    doc_prompt="doc: {text}",
    ...
)
```

**After (Option 2 - Use column_rename_mapping):**
```python
# Keep old dataset with "text" field - just remap it!
ds = [{"text": "query", "positive": "pos"}]
train_ds = Dataset.from_list(ds)

# Use column_rename_mapping to adapt old data
angle.fit(
    train_ds=train_ds,
    column_rename_mapping={"text": "query"},  # Automatic remapping!
    query_prompt="query: {text}",
    doc_prompt="doc: {text}",
    ...
)
```

**Command-line example:**
```bash
# With --column_rename_mapping, no need to modify dataset!
accelerate launch -m angle_emb.angle_trainer \
  --train_name_or_path your-old-dataset \
  --column_rename_mapping "text:query" \
  --query_prompt "query: {text}" \
  --doc_prompt "doc: {text}" \
  ...
```

### Scenario 3: Training LLM Model

**Before:**
```bash
# is_llm was auto-detected
torchrun -m angle_emb.angle_trainer \
--model_name_or_path NousResearch/Llama-2-7b-hf \
--apply_lora 1 \
...
```

**After:**
```bash
# Must explicitly set is_llm
accelerate launch -m angle_emb.angle_trainer \
--model_name_or_path NousResearch/Llama-2-7b-hf \
--is_llm 1 \
--apply_lora 1 \
...
```

---

## 🆘 Getting Help

If you encounter issues during migration:

1. Check the [updated documentation](https://angle.readthedocs.io/)
2. Review the [examples](https://github.com/SeanLee97/AnglE/tree/main/examples)
3. Open an issue on [GitHub](https://github.com/SeanLee97/AnglE/issues)

---

## 📦 Benefits of Upgrading

- **Simpler API**: No need to pre-tokenize datasets
- **More flexible**: Support for list-based sampling in Format B/C
- **Better prompting**: Separate prompts for queries and documents
- **Clearer configuration**: Explicit `is_llm` setting
- **Modern training**: `accelerate launch` support out of the box
- **Backward compatibility**: Use `--column_rename_mapping` to adapt old datasets without modification

Happy upgrading! 🚀

---

## 🔧 Quick Reference: column_rename_mapping

The `--column_rename_mapping` / `column_rename_mapping` parameter makes migration easier by allowing you to remap old column names to new ones without modifying your dataset.

**Syntax:**
- Command-line: `--column_rename_mapping "old_name:new_name"`
- Python: `column_rename_mapping={"old_name": "new_name"}`

**Common use cases:**
```bash
# Remap "text" to "query" for Format B/C datasets
--column_rename_mapping "text:query"

# Multiple remappings (separate with semicolons in CLI)
--column_rename_mapping "text:query;sentence1:text1;sentence2:text2"
```

**Python example:**
```python
angle.fit(
    train_ds=old_dataset,
    column_rename_mapping={
        "text": "query",
        "sentence1": "text1",
        "sentence2": "text2"
    },
    ...
)
```

This allows you to use existing datasets without any preprocessing!

