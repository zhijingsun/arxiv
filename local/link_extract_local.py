import os
import logging
import re
from datetime import datetime
from PyPDF2 import PdfReader
from local.crawl_arxiv_download import crawl_arxiv_and_save_pdf
# from assistant import dataset_assistant
from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.ollama import Ollama
from textwrap import dedent
from phi.llm.groq import Groq
from link_analysis import process_github_readme

model: str = "llama3-70b-8192"

def read_pdf_first_page(file_path: str) -> str:
    """读取PDF文件的第一页并返回其内容"""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        first_page = reader.pages[0]
        return first_page.extract_text()

def read_pdf(file_path: str) -> str:
    """读取完整PDF文件并返回其内容"""
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def extract_https_links(text: str) -> list:
    """从文本中提取所有HTTPS链接"""
    return re.findall(r'https://\S+', text)

def process_pdf_files(directory: str):
    """处理指定目录中的所有PDF文件"""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(directory, filename)
            pdf_id = filename.rsplit('.', 1)[0].replace('.', '')
            print(f"Processing file: {pdf_file_path}")
            print(f"PDF ID: {pdf_id}")
            pdf_content = read_pdf(pdf_file_path)
            # 读取PDF文件第一页内容
            pdf_content_first = read_pdf_first_page(pdf_file_path)
            query = f"""Tell me whether this paper, "{pdf_content_first}", create new dataset or not, in the format:
            Dataset: Yes.
            Dataset: No."""
            dataset_assistant = Assistant(
                # llm=Ollama(model="phi3:latest"),
                # llm=Ollama(model="llama3"),
                llm=Groq(model=model),
                instructions=[
                # "You will be provided an arxiv paper with pdf format.",
                # "You will analyze this document and write a report."
                f"""Tell me whether this paper, "{pdf_content_first}", create new dataset or not, in the format:
                    Dataset: Yes.
                    Dataset: No."""
                ],
                markdown=True
            )

            try:

                # 提取HTTPS链接
                https_links = extract_https_links(pdf_content_first)
                # 过滤掉包含 'doi.org' 的链接
                filtered_links = [link for link in https_links if 'doi.org' not in link]
                links_text = ', '.join(filtered_links) if filtered_links else "None"

                # 使用 assistant 检查是否创建了新的数据集
                # pdf_content = read_pdf(pdf_file_path)
                # query = f"""T
                # tell me whether this paper, "{pdf_content}", create new dataset or not, in the format:
                # Dataset: Yes.
                # Dataset: No."""
                # response = dataset_assistant.convert_response_to_string
                response = dataset_assistant.get_response(query)
                print(f"Assistant response: {response}")
                print(f"Filtered links: {filtered_links}")
                print(f"Links text: {links_text}")

                for i in filtered_links:
                    process_github_readme(i)
            except Exception as e:
                logging.error(f"An error occurred with file {pdf_file_path}: {str(e)}")

def process_all_subdirectories(base_directory: str):
    """处理基目录中的所有子目录"""
    subdirectories = ['IR', 'DB', 'AI', 'CL', 'CV', 'MA']
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(base_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            print(f"Processing directory: {subdirectory_path}")
            process_pdf_files(subdirectory_path)

def get_latest_date_directory(base_path: str) -> str:
    """获取路径中最新日期的文件夹"""
    date_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    date_folders.sort(key=lambda date: datetime.strptime(date, "%d_%b_%Y"), reverse=True)
    latest_directory = os.path.join(base_path, date_folders[0]) if date_folders else None
    return latest_directory

# 运行爬虫并保存PDF文件
crawl_arxiv_and_save_pdf()

# 获取最新日期的文件夹路径
base_path = "/Users/zhijingsun/Desktop/arxiv_paper_update"
latest_date_directory_path = get_latest_date_directory(base_path)

if latest_date_directory_path:
    print(f"Processing the latest directory: {latest_date_directory_path}")
    process_all_subdirectories(latest_date_directory_path)
else:
    print("No date directories found.")


