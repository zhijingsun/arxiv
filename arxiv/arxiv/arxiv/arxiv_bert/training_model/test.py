import os
import logging
import torch
import random
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed, BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

model_path = '/Users/zhijingsun/Desktop/东理/基础模型/fine_tuned_model'
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set seed for reproducibility
def set_all_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

set_all_seeds(42)

# 从JSON文件中读取数据
with open('/Users/zhijingsun/Desktop/东理/arxiv_bert/training_data/original/training_data.json', 'r') as f: 
    data = json.load(f)

for item in data:
    if 'url' in item:
        del item['url']
    item['content'] = item["title"] + ' ' + item["abstract"]
    del item['title']
    del item['abstract']

# Custom Dataset class
class PDFDataset(Dataset):
    def __init__(self, data, tokenizer, shuffle=False):
        self.data = data
        self.tokenizer = tokenizer
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["content"]
        label = item["label"]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512, add_special_tokens=True)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['label'] = torch.tensor(label, dtype=torch.long)
        return item

    def shuffle_data(self):
        if self.shuffle:
            import random
            random.shuffle(self.data)

    def get_label_distribution(self):
        labels = [item['label'] for item in self.data]
        positive = labels.count(1)
        negative = labels.count(0)
        return positive, negative

# Process data
processed_data = []
for item in data:
    try:
        if item['content'].strip():
            item['label'] = int(item['label'])
            item['content'] = item['content'].strip()
            processed_data.append(item)
    except ValueError:
        print(f"Invalid label: {item['label']} - Skipping this entry")

# Split data into training and testing sets
train_data, temp_data = train_test_split(processed_data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

test_dataset = PDFDataset(test_data, tokenizer, shuffle=False)


def test_model(test_dataset, tokenizer, model_save_path, epoch):
    # Load the trained model
    model = BertForSequenceClassification.from_pretrained(model_save_path)

    # Define device - GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluate the model on test set
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    all_preds = []
    all_labels = []

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Predictions and labels
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

    print(epoch)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_model(test_dataset, tokenizer, model_path, "fine_tune")

   