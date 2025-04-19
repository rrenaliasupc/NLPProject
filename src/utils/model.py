import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizerFast, BertModel, BertPreTrainedModel

from pathlib import Path
# Go two levels up from this file's directory: /src/utils/ → / → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DEFAULT_PATH = f"{PROJECT_ROOT}/build/model.pt"
DATASET_DEFAULT_PATH = f"{PROJECT_ROOT}/dataset/dataset_augmented_merged.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = [
    "O", 
    "B-ACTION", "I-ACTION",
    "B-ACTION-TURN-ON", "I-ACTION-TURN-ON",
    "B-ACTION-TURN-OFF", "I-ACTION-TURN-OFF",
    "B-DEVICE", "I-DEVICE",
    "B-LOCATION", "I-LOCATION",
    "B-ACTION-NONE", "I-ACTION-NONE",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

class SmartHomeNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Flatten to calculate loss
            loss = loss_fn(logits.view(-1, len(LABELS)), labels.view(-1))
            return loss, logits
        return logits

def infer(model_path, text):
    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    config = BertConfig.from_pretrained("bert-base-multilingual-cased", num_labels=len(ID2LABEL))
    model = SmartHomeNER.from_pretrained("bert-base-multilingual-cased", config=config)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    words = text.split()
    tokens = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding='max_length', max_length=32)
    word_ids = tokens.word_ids()

    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    # Reconstruct prediction per word
    entities = {}
    current_entity = None
    current_type = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        label = ID2LABEL[predictions[i]]
        word = words[word_idx]

        if label.startswith("B-"):
            current_entity = word
            current_type = label[2:]
        elif label.startswith("I-") and current_entity and current_type == label[2:]:
            current_entity += " " + word
        else:
            if current_entity:
                entities[current_type.lower()] = current_entity
            current_entity = None
            current_type = None

    # Catch the last one
    if current_entity:
        entities[current_type.lower()] = current_entity

    return entities