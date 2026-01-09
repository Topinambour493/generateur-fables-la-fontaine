import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# ==============================
# ENV
# ==============================
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATASET_PATH = "fables_lora.jsonl"
OUTPUT_DIR = "./tinyllama_fables_lora"

MAX_LENGTH = 512
EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUM = 2
LEARNING_RATE = 2e-4

# ==============================
# DATASET
# ==============================
dataset = load_dataset("json", data_files=DATASET_PATH)["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    texts = [
        p + c for p, c in zip(batch["prompt"], batch["completion"])
    ]
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH
    )

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenisation"
)

dataset.set_format("torch")

# ==============================
# QUANTIZATION
# ==============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ==============================
# MODEL
# ==============================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)

# ==============================
# LoRA
# ==============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==============================
# TRAINING
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHSPOCHS if False else EPOCHS,  # sécurité IDE
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

print("🚀 Début entraînement LoRA...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Entraînement terminé")
