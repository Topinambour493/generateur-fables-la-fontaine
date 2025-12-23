# train_lora.py
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# =====================
# CONFIG
# =====================
MODEL_NAME = "bigscience/bloom-560m"
DATA_PATH = "corpus.jsonl"
OUTPUT_DIR = "./lora-la-fontaine-bloom"
MAX_LENGTH = 512

# =====================
# MODEL & TOKENIZER
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

# =====================
# LORA CONFIG
# =====================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query_key_value"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =====================
# DATASET
# =====================
dataset = load_dataset("json", data_files=DATA_PATH)

def format_example(e):
    return f"""### Instruction:
{e['prompt']}
### Réponse:
{e['output']}
"""

def tokenize(example):
    text = format_example(example)
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(
    tokenize,
    remove_columns=dataset["train"].column_names
)

# =====================
# TRAINING
# =====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()

# =====================
# SAVE LORA
# =====================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ LoRA sauvegardé dans {OUTPUT_DIR}")
