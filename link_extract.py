import os
import logging
import re
import requests
from datetime import datetime
from PyPDF2 import PdfReader
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from database import add_dataset_info #
from abstract_analysis import analyse_content #

from link_valid import process_content
from crawl_arxiv_url import get_pdf_urls

# 设置日志配置
# logging.basicConfig(level=logging.INFO)

# 设置模型路径和分词器
MODEL_PATH = os.path.expanduser('~/Desktop/东理/BERT/fine_tuned_model')
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

def read_pdf_from_url(url: str) -> str:
    """读取在线PDF文件并返回其内容"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open('/tmp/temp.pdf', 'wb') as f:
            f.write(response.content)
        return read_pdf('/tmp/temp.pdf')
    except requests.RequestException as e:
        logging.error(f"Failed to fetch PDF from URL: {url} - {e}")
        raise
    
def read_pdf(file_path: str) -> str:
    """读取PDF文件并返回其内容"""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
            return content
    except Exception as e:
        logging.error(f"Failed to read PDF: {file_path} - {e}")
        raise

def read_pdf_first_page_from_url(url: str) -> str:
    """读取在线PDF文件的第一页并返回其内容"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open('/tmp/temp.pdf', 'wb') as f:
            f.write(response.content)
        return read_pdf_first_page('/tmp/temp.pdf')
    except requests.RequestException as e:
        logging.error(f"Failed to fetch PDF from URL: {url} - {e}")
        raise

def read_pdf_first_page(file_path: str) -> str:
    """读取PDF文件的第一页并返回其内容"""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            first_page = reader.pages[0]
            return first_page.extract_text()
    except Exception as e:
        logging.error(f"Failed to read first page of PDF: {file_path} - {e}")
        raise

def extract_https_links(text: str) -> list:
    """从文本中提取所有HTTPS链接,并去除'arXiv:'部分"""
    # 提取所有的 HTTPS 链接
    all_links = re.findall(r'https://\S+', text)
    # 去除每个链接中包含的 'arXiv:' 部分
    cleaned_links = [re.sub(r'arXiv:\S+', '', link).strip() for link in all_links]
    return cleaned_links

def extract_link_context(text: str, link: str, window_size: int = 100) -> str:
    """提取链接前后的文本内容"""
    # 找到链接在文本中的位置
    index = text.find(link)
    if index == -1:
        return "Link not found in text."

    # 截取链接前后指定窗口大小的文本
    start_index = max(0, index - window_size)
    end_index = min(len(text), index + len(link) + window_size)
    context = text[start_index:end_index]

    # 标记链接的位置
    link_start = index - start_index
    link_end = link_start + len(link)
    context = context[:link_start] + " " + context[link_start:link_end] + " " + context[link_end:]

    return context


def classify_text(text: str) -> str:
    """使用BERT模型分类文本, 判断是否包含数据库"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return "Dataset: Yes" if predictions.item() == 1 else "Dataset: No"

def process_pdf_from_url(pdf_data):
    """处理在线PDF文件"""
    url = str(pdf_data['pdf_link'])
    pdf_id = str(pdf_data['pdf_id'])
    # print(pdf_id)
    # print(url)
    try:
        pdf_content_first = read_pdf_first_page_from_url(url)
        pdf_content = read_pdf_from_url(url)
        
        
        # 提取链接上下文并用于分类文本
        for link in extract_https_links(pdf_content_first):
            if 'doi.org' in link:
                continue
            
            context = extract_link_context(pdf_content_first, link) # 链接前后的文本内容
            classification_result = classify_text(context)
            print(classification_result)
            logging.info(f"Classification result: {classification_result}")

            # 处理分类结果
            if classification_result == "Dataset: Yes":
                encoded_url = process_content(pdf_data, link) # dataset url
                json_result = analyse_content(pdf_data, context, url) 
                add_dataset_info(pdf_data['category'], pdf_id, json_result) #
    except Exception as e:
        logging.error(f"An error occurred with URL {url}: {str(e)}")




# 示例调用
if __name__ == "__main__":
    pdf_urls = get_pdf_urls()
    for pdf_data in pdf_urls:
        process_pdf_from_url(pdf_data)
