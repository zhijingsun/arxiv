from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
import json
import torch
from torch.utils.data import Dataset

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# 加载预训练模型和分词器
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 加载数据集并进行分词
def load_and_tokenize_data(file_path, tokenizer):
    # 读取 json 数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取 instruction 和 input 字段，并组合成文本
    texts = [f"{item['instruction']} {item['input']}" for item in data]
    
    # 进行分词
    tokenized_data = tokenizer(texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

    return tokenized_data

# 加载数据并进行分词
tokenized_data = load_and_tokenize_data('data.json', tokenizer)

# 使用 train_test_split 划分数据集
train_inputs, eval_inputs, train_labels, eval_labels = train_test_split(
    tokenized_data['input_ids'], tokenized_data['attention_mask'], test_size=0.2, random_state=42)

# 构建 PyTorch 数据集
train_dataset = TextDataset({'input_ids': train_inputs, 'attention_mask': train_labels})
eval_dataset = TextDataset({'input_ids': eval_inputs, 'attention_mask': eval_labels})

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 微调模型
trainer.train()
