import re
import unicodedata
import requests
import json

def get_prompt(title):
    return "Compose une fable en vers, dans le style de La Fontaine. Titre imposé : " +  title

def without_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def append_in_jsonl(title, fable):
    with open("corpus.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": get_prompt(title), "output": fable}, ensure_ascii=False) + "\n")


def get_fable(title):
    text_norm = without_accents(text)
    title_norm = without_accents(title)

    #gestion du cas où le titre est suivi d'un . ex: TITRE.
    pattern = rf"{re.escape(title_norm)}\.\s*\n([\s\S]*?)\[\s*illustration\s*\]"
    match = re.search(pattern, text_norm, re.IGNORECASE)

    if not match:
        # gestion du cas où le titre est suivi d'un [nombre]. ex: TITRE[quelque chose].
        pattern = rf"{re.escape(title_norm)}\s*\[[^\]]+\]\.\s*\n([\s\S]*?)\[\s*Illustration\s*\]"
        match = re.search(pattern, text_norm, re.IGNORECASE)
        if not match:
            return None

        # récupérer le texte original (non normalisé)
        start, end = match.span(1)
        return text[start:end].strip()

    # récupérer le texte original (non normalisé)
    start, end = match.span(1)
    return text[start:end].strip()


def clean_titles(titles):
    cleaned = []
    for t in titles:
        if t == "Le Dragon à plusieurs têtes et le Dragon à plusieurs queues":
            cleaned.append("Le Dragon à plusieurs têtes, et le Dragon à plusieurs queues")
            continue

        if t == "La Génisse, la Chèvre et la Brebis en société avec le Lion":
            cleaned.append("La Génisse, la Chèvre et la Brebis, en société avec le Lion")
            continue
        if t == "Le Charretier embourbé":
            cleaned.append("Le Chartier embourbé")
            continue
        cleaned.append(t)
    return cleaned

def get_titles(text):
    pattern = r"([A-ZLÉÈÀÂÊÎÔÛÇ][^\.\n]*?(?:\n\s+[^\.\n]+?)?)\.\s+\d+"
    matches = re.findall(pattern, text)

    # Nettoyage des retours à la ligne et espaces
    titles = [
        " ".join(m.split()) for m in matches
    ]


    return clean_titles(titles)

def get_names_fables():
    fables_table_matiere = text.split("PAGES.")[1].split("PARIS.--J. CLAYE, IMPRIMEUR, RUE SAINT-BENOIT, 7.")[0]
    for name_fable in get_titles(fables_table_matiere):
        fable = get_fable(name_fable)
        if fable is not None:
            append_in_jsonl(name_fable, fable)


compilation = "https://www.gutenberg.org/cache/epub/56327/pg56327.txt"
text = requests.get(compilation).text
fables = text.split("PAGES.")[0]


get_names_fables()