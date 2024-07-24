import os
import logging
import re
import requests
from datetime import datetime
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertForSequenceClassification

# 设置日志配置
# logging.basicConfig(level=logging.INFO)

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

def extract_https_links(text: str) -> list:
    """Extract all HTTPS links from the text and clean 'arXiv:' part if present."""
    all_links = re.findall(r'https://\S+', text)
    cleaned_links = []
    for link in all_links:
        # Remove 'arXiv:' part
        cleaned_link = re.sub(r'arXiv:\S+', '', link).strip()
        # Find the position of the link in the original text
        start_index = text.find(link) + len(link)
        # Check the character immediately after the link
        if start_index < len(text) and text[start_index+1] not in ['.', ',', '1']:
            # Extract the next word and append it to the link
            remaining_text = text[start_index:].strip()
            next_word = re.match(r'\S+', remaining_text)
            if next_word: #if match the format of string + white space
                extended_link = cleaned_link + next_word.group(0)
                # Validate the extended link
                response = requests.get(extended_link, stream=True)
                if response.status_code != 404:
                    cleaned_links.append(extended_link)
                    continue
        # If the link is followed by punctuation or the extended link is invalid, use the original link
        if cleaned_link.endswith('.') or cleaned_link.endswith(','):
            cleaned_link = cleaned_link[:-1]
        cleaned_links.append(cleaned_link)
    
    return cleaned_links