# generate_fable.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# =====================
# CONFIG
# =====================
MODEL_NAME = "bigscience/bloom-560m"
LORA_PATH = "./lora-la-fontaine-bloom"

# =====================
# LOAD MODEL
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# =====================
# PROMPT
# =====================
prompt = """### Instruction:
Écris une fable en vers dans le style de La Fontaine.

### Titre:
La Cigale et le Fromage

### Réponse:
"""

# =====================
# GENERATION
# =====================
result = pipe(
    prompt,
    max_new_tokens=250,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1
)

print("\n📜 FABLE GÉNÉRÉE :\n")
print(result[0]["generated_text"])
