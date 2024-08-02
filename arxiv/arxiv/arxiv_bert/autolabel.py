import os
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
import json
import pandas as pd

# Define a custom dataset class
class PDFDataset(Dataset):
    def __init__(self, papers, tokenizer):
        self.papers = papers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        paper = self.papers[idx]
        encoding = self.tokenizer.encode_plus(
            paper['abstract'],
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'url': paper['url'],
            'title': paper['title'],
            'abstract': paper['abstract']
        }

def predict(test_dataset, model_save_path):
    # Load the trained model
    model = BertForSequenceClassification.from_pretrained(model_save_path)

    # Define device - GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluate the model on test set
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    predictions = []

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        
        for i in range(len(preds)):
            predictions.append({
                'url': batch['url'][i],
                'title': batch['title'][i],
                'abstract': batch['abstract'][i],
                'predicted_label': int(preds[i])
            })

    return predictions

def json_to_excel(json_file, excel_file):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 将 JSON 数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存为 Excel 文件
    df.to_excel(excel_file, index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 获取脚本文件的当前路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构造相对路径
    json_file_path = os.path.join(script_dir, '../../latest_papers.json')

    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        papers = json.load(f)
    
    # Path to the fine-tuned model
    model_save_path = '/Users/zhijingsun/Desktop/东理/fine_tuned_model'

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_save_path)

    # Load test dataset
    test_dataset = PDFDataset(papers, tokenizer)

    # Predict the labels
    predictions = predict(test_dataset, model_save_path)

    # Save predictions to JSON file
    predictions_json_file = 'predictions.json'
    with open(predictions_json_file, 'w') as f:
        json.dump(predictions, f, indent=4)

    # Convert JSON file to Excel file
    predictions_excel_file = 'predictions.xlsx'
    json_to_excel(predictions_json_file, predictions_excel_file)

    print(f"转换完成：{predictions_json_file} -> {predictions_excel_file}")
