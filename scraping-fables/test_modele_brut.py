from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =====================
# CONFIG
# =====================
MODEL_NAME = "bigscience/bloom-560m"

# =====================
# LOAD MODEL
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# =====================
# PROMPT
# =====================
prompt = "Compose une fable en vers, dans le style de La Fontaine. Titre imposé : L'Aigle et la Pie"

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

print("\n📜 SORTIE DU MODÈLE BRUT :\n")
print(result)
print(result[0]["generated_text"])
