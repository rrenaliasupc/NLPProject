import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SmartHomeClassifier(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.action_head = nn.Linear(hidden_size, 3)
        self.device_head = nn.Linear(hidden_size, 4)
        self.location_head = nn.Linear(hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.pooler_output
        return {
            "action": self.action_head(cls),
            "device": self.device_head(cls),
            "location": self.location_head(cls)
        }

def load_model(path):
    my_model = SmartHomeClassifier()
    my_model.load_state_dict(torch.load(path))
    my_model.eval()
    return my_model

def tokenize(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True)

def infer(model_path, text):
    my_model = load_model(model_path)
    tokens = tokenize(text)

    with torch.no_grad():
        outputs = my_model(tokens["input_ids"], tokens["attention_mask"])

    return outputs
