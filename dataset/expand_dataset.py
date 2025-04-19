import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ORIG_PATH = f"{PROJECT_ROOT}/dataset/dataset_llm.jsonl"
DATASET_AUG_PATH = f"{PROJECT_ROOT}/dataset/dataset_augmented.jsonl"

def stt_typo(token):
    """Apply STT-like distortions to a given token."""
    typos = {
        "llum": ["llom", "jun", "llúm"],
        "menjador": ["mejador", "menyador", "mejado"],
        "cuina": ["quina", "cunia", "cuin"],
        "passadís": ["passadi", "pasadís"],
        "habitació": ["avitació", "habitacio", "habitación"],
        "televisió": ["tele", "telebició", "telebisio"],
        "porta": ["porra", "pòrta", "porta"],
        "finestra": ["finesa", "finestrà", "finetra"],
        "persiana": ["persia", "perxiana", "persian"],
        "obrir": ["obrir", "ubrir", "obrí"],
        "apagar": ["apagar", "apga", "pagar"],
        "obriu": ["obreu", "obri", "obriu"],
        "tanqueu": ["tanque", "tanquiu", "tanqueu"],
        "enceneu": ["enceneu", "enseneu", "encen"],
        "engega": ["engega", "engeja", "engeba"],
        "posa": ["posa", "possa", "pussa"],
        "tanca": ["tanca", "tanka", "tanque"],
        "apaga": ["apaga", "apague", "apga"],
        "obrir": ["ubrir", "obrí", "obir"],
    }

    drop_probs = {
        "el": 0.4, "la": 0.4, "l": 0.3, "de": 0.3,
        "del": 0.4, "que": 0.3, "d'": 0.4, "l'": 0.4,
    }

    # drop some short function words
    if token.lower() in drop_probs and random.random() < drop_probs[token.lower()]:
        return ""

    # apply fuzzy substitution
    lower = token.lower()
    if lower in typos and random.random() < 0.6:
        return random.choice(typos[lower])

    # apostrophe confusion
    if re.match(r"\w+'\w+", token):
        return token.replace("'", "")

    return token

def tokenize_bio(text, labels):
    tokens = text.strip().split()

    # If lengths match, we're good
    if len(tokens) == len(labels):
        return tokens, labels

    # Otherwise try splitting apostrophes
    split_tokens = []
    split_labels = []
    for token, label in zip(tokens, labels):
        if "'" in token:
            subtokens = token.split("'")
            if len(subtokens) == 2:
                # e.g., "l'habitació" → ["l", "habitació"]
                split_tokens.extend([subtokens[0], subtokens[1]])
                split_labels.extend([label, label])  # duplicate label for both parts
            else:
                # Unexpected structure, keep whole token
                split_tokens.append(token)
                split_labels.append(label)
        else:
            split_tokens.append(token)
            split_labels.append(label)

    if len(split_tokens) != len(split_labels):
        raise ValueError(f"Token-label mismatch after apostrophe splitting:\nTokens: {split_tokens}\nLabels: {split_labels}")

    return split_tokens, split_labels


def apply_stt_typo_to_sentence(entry):
    """Takes one dataset entry and returns a distorted version with STT typos."""
    original_text = entry["text"]
    labels = entry["labels"]

    tokens, labels = tokenize_bio(original_text, labels)

    distorted_tokens = []
    distorted_labels = []

    for token, label in zip(tokens, labels):
        typo_token = stt_typo(token)
        if typo_token == "":
            # drop word and label
            continue
        distorted_tokens.append(typo_token)
        distorted_labels.append(label)

    if not distorted_tokens:
        return None  # skip empty outputs

    return {
        "text": " ".join(distorted_tokens),
        "labels": distorted_labels
    }

def augment_jsonl_with_typos(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            original = json.loads(line)
            json.dump(original, outfile, ensure_ascii=False)
            outfile.write("\n")

            augmented = apply_stt_typo_to_sentence(original)
            if augmented and augmented != original:
                json.dump(augmented, outfile, ensure_ascii=False)
                outfile.write("\n")

if __name__ == "__main__":
    augment_jsonl_with_typos(DATASET_ORIG_PATH, DATASET_AUG_PATH)
