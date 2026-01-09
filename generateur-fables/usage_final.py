import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
from peft import PeftModel
import time

# ==============================
# ENV
# ==============================
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_BASE = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_LORA = "./tinyllama_fables_lora"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ==============================
# LOAD TOKENIZER
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_LORA)
tokenizer.pad_token = tokenizer.eos_token

# ==============================
# LOAD MODEL + LORA
# ==============================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    torch_dtype=torch.float16,
    device_map=None
).to("cuda")

model = PeftModel.from_pretrained(model, MODEL_LORA)
model.eval()

# ==============================
# GENERATION
# ==============================
title = "La cigale et la fourmi"

prompt = (
    "Écris une fable à la manière de Jean de La Fontaine.\n"
    "Respecte un style classique, narratif, avec une morale.\n"
    f"Titre : {title}\n\nFable :\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")



start = time.time()
streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

generation_kwargs = dict(
    **inputs,
    max_new_tokens=400,
    do_sample=True,
    temperature=0.6,
    top_p=0.8,
    streamer=streamer,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
    eos_token_id=tokenizer.eos_token_id,
)

generated_text = []

thread = threading.Thread(
    target=model.generate,
    kwargs=generation_kwargs
)
thread.start()

print("\n===== FABLE EN COURS =====\n")

for chunk in streamer:
    print(chunk, end="", flush=True)
    generated_text.append(chunk)

thread.join()  # on attend la fin complète

final_text = "".join(generated_text)

print("\n\n===== TEXTE COMPLET =====\n")
print(final_text)
print("Temps:", time.time() - start)