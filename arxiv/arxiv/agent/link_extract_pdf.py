import logging
import re
import requests
from PyPDF2 import PdfReader

def read_pdf_from_url(url: str) -> str:
    """读取在线PDF文件并返回其内容"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_path = '/tmp/temp.pdf'  # 存储临时PDF的路径
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return read_pdf(pdf_path)
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
    """提取所有HTTPS链接并清除'arXiv:'前缀（如果存在）"""
    # 找到 references 部分的起始位置
    reference_start = re.search(r'(References|Bibliography)', text, re.IGNORECASE)
    if reference_start:
        text = text[:reference_start.start()]

    all_links = re.findall(r'https://\S+', text)
    cleaned_links = []
    for link in all_links:
        # 清理 'arXiv:' 前缀
        cleaned_link = re.sub(r'arXiv:\S+', '', link).strip()
        # 如果链接后有字符，不是结束符，尝试扩展链接
        start_index = text.find(link) + len(link)
        if start_index < len(text) and text[start_index] not in ['.', ',', '1']:
            remaining_text = text[start_index:].strip()
            next_word = re.match(r'\S+', remaining_text)
            if next_word:
                extended_link = cleaned_link + next_word.group(0)
                # 验证扩展链接
                try:
                    response = requests.get(extended_link, stream=True)
                    if response.status_code != 404:
                        cleaned_links.append(extended_link)
                        continue
                except requests.RequestException:
                    pass
        # 如果链接以句号或逗号结尾，去掉它
        if cleaned_link.endswith('.') or cleaned_link.endswith(','):
            cleaned_link = cleaned_link[:-1]
        cleaned_links.append(cleaned_link)
    
    return cleaned_links

# def main():
#     arxiv_pdf_url = "https://arxiv.org/pdf/2406.16048"
#     pdf_text = read_pdf_from_url(arxiv_pdf_url)
#     links = extract_https_links(pdf_text)
#     print("Extracted links:")
#     for link in links:
#         print(link)

# if __name__ == '__main__':
#     main()
