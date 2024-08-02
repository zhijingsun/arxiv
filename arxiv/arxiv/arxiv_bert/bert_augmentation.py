import os
import logging
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# 禁用 wandb
os.environ["WANDB_DISABLED"] = "true"

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
with open('/Users/zhijingsun/Desktop/东理/arxiv/arxiv/arxiv_bert/training_data/augmentation_data/240731.json', 'r') as f: 
    data = json.load(f)

for item in data:
    if 'url' in item:
        del item['url']
    item['content'] = item["title"] + ' ' + item["abstract"]
    del item['title']
    del item['abstract']

# Set logging configuration
logging.basicConfig(level=logging.INFO)

# Define model and tokenizer
MODEL_NAME = "/Users/zhijingsun/Desktop/东理/fine_tuned_model" # 之前训练好的模型
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


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
train_data, valid_data = train_test_split(processed_data, test_size=0.2, random_state=42, stratify=[item['label'] for item in processed_data])

# Create datasets
train_dataset = PDFDataset(train_data, tokenizer, shuffle=True)
valid_dataset = PDFDataset(valid_data, tokenizer, shuffle=False)

# Log label distribution
train_pos, train_neg = train_dataset.get_label_distribution()
valid_pos, valid_neg = valid_dataset.get_label_distribution()

print(f"Training set - Positive: {train_pos}, Negative: {train_neg}")
print(f"Validation set - Positive: {valid_pos}, Negative: {valid_neg}")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train_model():
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=2,
        logging_dir='./logs',
        logging_steps=5,
        evaluation_strategy="steps",  # Evaluate at each step
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=5e-5,  
        weight_decay=0.01,
    )

    # Early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
        # callbacks=[early_stopping_callback, save_best_model_callback]
    )

    # Start training
    trainer.train()

    # # Create directory to save the model
    # model_save_path = os.path.expanduser(MODEL_NAME)
    # os.makedirs(model_save_path, exist_ok=True)

    # Save the model and tokenizer

    model.save_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_NAME)

# Train the model
if __name__ == "__main__":
    train_model()

    model_save_path = os.path.expanduser('~/Desktop/东理/fine_tuned_model')
