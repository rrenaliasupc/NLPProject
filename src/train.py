import torch
from transformers import BertTokenizer
from torch.optim import AdamW
from utils.model import SmartHomeClassifier, MODEL_DEFAULT_PATH, DATASET_DEFAULT_PATH
from utils.labels import *
import json

# Load data
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
with open(DATASET_DEFAULT_PATH) as f:
    samples = [json.loads(l) for l in f]

X = tokenizer([s["text"] for s in samples], padding=True, truncation=True, return_tensors="pt")
y_action = torch.tensor([label_to_id(s["action"], ACTIONS) for s in samples])
y_device = torch.tensor([label_to_id(s["device"], DEVICES) for s in samples])
y_location = torch.tensor([label_to_id(s["location"], LOCATIONS) for s in samples])

# Model
model = SmartHomeClassifier()
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(5):
    outputs = model(X["input_ids"], X["attention_mask"])
    loss = (
        loss_fn(outputs["action"], y_action) +
        loss_fn(outputs["device"], y_device) +
        loss_fn(outputs["location"], y_location)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), MODEL_DEFAULT_PATH)
