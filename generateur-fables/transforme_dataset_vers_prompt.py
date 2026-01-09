import json

INPUT = "corpus.jsonl"
OUTPUT = "fables_lora.jsonl"


def build_prompt(title):
    return (
        "Écris une fable à la manière de Jean de La Fontaine.\n"
        "Respecte un style classique, narratif, avec une morale.\n"
        f"Titre : {title}\n\nFable :\n"
    )


with open(INPUT, "r", encoding="utf-8") as f_in, \
        open(OUTPUT, "w", encoding="utf-8") as f_out:
    for line in f_in:
        item = json.loads(line)
        prompt = build_prompt(item["titre"])
        completion = item["texte"].strip()

        record = {
            "prompt": prompt,
            "completion": completion
        }

        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Dataset prêt :", OUTPUT)
