import torch
from transformers import BertTokenizerFast as BertTokenizer, BertConfig
from torch.optim import AdamW
from utils.model import SmartHomeNER, LABEL2ID, MODEL_DEFAULT_PATH, DATASET_DEFAULT_PATH, DEVICE
import json, random

BATCH_SIZE = 70
EPOCHS = 5

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# --- Dataset format: each entry has "text" and "labels" (list of BIO tags per token)
def encode_sample(sample):
    tokens = tokenizer(sample["text"].split(), is_split_into_words=True, return_tensors="pt", truncation=True, padding='max_length', max_length=32)
    word_ids = tokens.word_ids()  # mapping of subword to word index

    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label = sample["labels"][word_idx]
            label_ids.append(LABEL2ID[label])
    tokens["labels"] = torch.tensor([label_ids])
    return tokens

with open(DATASET_DEFAULT_PATH) as f:
    raw_samples = [json.loads(l) for l in f]
    encoded = [encode_sample(s) for s in raw_samples]

random.shuffle(encoded)
train_size = int(0.8 * len(encoded))
test_size = len(encoded) - train_size

# Split into train/test
train_dataset, test_dataset = encoded[:train_size], encoded[train_size:]
assert len(test_dataset) == test_size

# Model
config = BertConfig.from_pretrained("bert-base-multilingual-cased", num_labels=len(LABEL2ID))
model = SmartHomeNER.from_pretrained("bert-base-multilingual-cased", config=config).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(EPOCHS):
    random.shuffle(train_dataset)
    current_batch = train_dataset[:BATCH_SIZE]
    # Stack everything
    input_ids = torch.cat([e["input_ids"] for e in current_batch])
    attention_mask = torch.cat([e["attention_mask"] for e in current_batch])
    labels = torch.cat([e["labels"] for e in current_batch])

    optimizer.zero_grad()
    input_ids_batch = input_ids.to(DEVICE)
    attention_mask_batch = attention_mask.to(DEVICE)
    labels_batch = labels.to(DEVICE)

    loss, _ = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), str(MODEL_DEFAULT_PATH))

# validation
random.shuffle(test_dataset)
current_batch = test_dataset[:BATCH_SIZE]
input_ids = torch.cat([e["input_ids"] for e in test_dataset])
attention_mask = torch.cat([e["attention_mask"] for e in test_dataset])
labels = torch.cat([e["labels"] for e in test_dataset])

model.eval()
with torch.no_grad():
    optimizer.zero_grad()
    input_ids_batch = input_ids.to(DEVICE)
    attention_mask_batch = attention_mask.to(DEVICE)
    labels_batch = labels.to(DEVICE)

    loss, _ = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)

print(f"Validation loss = {loss:.4f}")
