import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import re
from download import download_github_folder 

def extract_urls_from_arxiv(arxiv_url):

    part = arxiv_url.split('/')[3]
    if arxiv_url == 'pdf':
        # If the URL ends with ".pdf", skip processing
        return []
    response = requests.get(arxiv_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    urls = []

    bib_section = soup.find('section', {'id': 'bib'})
    if bib_section:
        bib_section.decompose()

    for article in soup.find_all('article'):
        for a in article.find_all('a', href=True):
            url = a['href']
            if a.text.strip() == url.strip() and url.startswith('http') and not url.startswith('https://arxiv.org/html'):
                urls.append(url)
    
    return urls

def find_datasets_in_github(url):
    datasets = []
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    for a in soup.find_all('a', href=True):
        file_url = a['href']
        if re.search(r'data|dataset', file_url, re.IGNORECASE) and file_url.startswith('/'):
            if not file_url.startswith('http'):
                file_url = 'https://github.com' + file_url
            datasets.append(file_url)

    datasets = list(set(datasets))

    return datasets

def main():
    st.title("arXiv Dataset Extractor")
    
    arxiv_url = st.text_input("Enter arXiv URL", "https://arxiv.org/html/2407.08948")

    if st.button("Process URL"):
        st.write(f"Processing {arxiv_url}")
        urls = extract_urls_from_arxiv(arxiv_url)
        st.write(f"Extracted {len(urls)} URLs")
        st.write(f"Extracted URLs: {urls}")

        all_datasets = []
        for url in urls:
            if 'github.com' in url:
                datasets = find_datasets_in_github(url)
                st.write(f"Found {len(datasets)} datasets in {url}")
                st.write(datasets)
                all_datasets.extend(datasets)
            elif 'huggingface.co' in url:
                # Add similar logic for handling datasets from HuggingFace
                pass
        
        if all_datasets:
            st.session_state.datasets = all_datasets
            st.session_state.download_initiated = False  # Reset download state
        else:
            st.write("No datasets found.")
    
    # Handle download state and dataset selection
    if 'datasets' in st.session_state and st.session_state.datasets:
        selected_datasets = st.multiselect("Select datasets to download", st.session_state.datasets)
        
        if selected_datasets:
            if st.button("Download Selected Datasets"):
                st.session_state.download_initiated = True
                for link in selected_datasets:
                    st.write("Start Downloading Selected Datasets")
                    st.write(f"link: {link}")
                    if link.startswith("https://github.com/"):
                        link = link[len("https://github.com/"):]
                        parts = link.split('/')
                        repo_url = f"{parts[0]}/{parts[1]}"
                        folder_path = '/'.join(parts[4:])
                        dest_dir = "/Users/zhijingsun/Downloads/sa-swin-datasets"
                        
                        # Ensure destination directory exists
                        if not os.path.exists(dest_dir):
                            os.makedirs(dest_dir)
                        
                        download_github_folder(repo_url, folder_path, dest_dir)
                        
                        st.write(f"Files downloaded to: {os.path.abspath(dest_dir)}")
                    else:
                        st.write("暂不支持下载该数据集")
        elif st.session_state.download_initiated:
            st.write("Download completed. You may now close this window.")

if __name__ == '__main__':
    main()
