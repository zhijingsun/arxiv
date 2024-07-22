import requests
from bs4 import BeautifulSoup
import os
import re
import wget
import json

def extract_urls_from_arxiv(arxiv_url):
    """
    从arXiv论文页面的 article 标签中提取所有URL
    """
    response = requests.get(arxiv_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    urls = []

    # 移除 bibliography section
    bib_section = soup.find('section', {'id': 'bib'})
    if (bib_section):
        bib_section.decompose()

    # 提取所有 article 标签中的 a 标签的 href 属性
    for article in soup.find_all('article'):
        for a in article.find_all('a', href=True):
            url = a['href']
            # 确保链接显示在网页上
            if a.text.strip() == url.strip() and url.startswith('http') and not url.startswith('https://arxiv.org/html'):
                urls.append(url)
    
    return urls

def find_datasets_in_github(url):
    """
    从GitHub仓库中查找数据集链接
    """
    datasets = []
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 查找所有可能的数据文件链接
    for a in soup.find_all('a', href=True):
        file_url = a['href']
        if re.search(r'data|dataset', file_url, re.IGNORECASE) and file_url.startswith('/'):
            # 确保链接是相对路径或完整的GitHub URL
            print(file_url)
            if not file_url.startswith('http'):
                file_url = 'https://github.com' + file_url
            datasets.append(file_url)

    datasets = list(set(datasets))

    return datasets


def download_files(file_urls, base_url, download_path='datasets'):
    """
    下载文件
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    for file_url in file_urls:
        # 构造完整URL
        if not file_url.startswith('http'):
            file_url = base_url + file_url
            print(f"file_url: {file_url}")
        print(f"file_url: {file_url}")
        # # 下载文件
        # filename = wget.download(file_url, out=download_path)
        # print(f"Downloaded {filename}")

def main():

    arxiv_url = "https://arxiv.org/html/2407.08948"
    print(f"Processing {arxiv_url}")
    urls = extract_urls_from_arxiv(arxiv_url)
    print(f"Extracted {len(urls)} URLs")
    print(f"Extracted URLs: {urls}")

    for url in urls:
        if 'github.com' in url:
            datasets = find_datasets_in_github(url)
            print(f"Found {len(datasets)} datasets in {url}")
            print(datasets)

            if datasets:
                base_url = '/'.join(url.split('/')[:5]) + '/'
                print(f"base url: {base_url}")
                download_files(datasets, base_url)
        elif 'huggingface.co' in url:
            # 这里可以添加类似的逻辑来处理 HuggingFace 上的数据集
            pass

    # # 读取 latest_papers.json 文件
    # with open('latest_papers.json', 'r') as f:
    #     papers = json.load(f)

    # for paper in papers:
    #     arxiv_url = paper['url']  # 假设 JSON 文件中每个对象有一个 'url' 字段  
    #     print(f"Processing {arxiv_url}")

    #     # 第一步：从 arXiv 页面提取所有 URL
    #     urls = extract_urls_from_arxiv(arxiv_url)
    #     print(f"Extracted {len(urls)} URLs")
    #     print(f"Extracted URLs: {urls}")

        # # 第二步：查找和下载数据集
        # for url in urls:
        #     if 'github.com' in url:
        #         datasets = find_datasets_in_github(url)
        #         print(f"Found {len(datasets)} datasets in {url}")
        #         if datasets:
        #             base_url = '/'.join(url.split('/')[:5]) + '/'
        #             download_files(datasets, base_url)
        #     elif 'huggingface.co' in url:
        #         # 这里可以添加类似的逻辑来处理 HuggingFace 上的数据集
        #         pass

if __name__ == '__main__':
    main()
