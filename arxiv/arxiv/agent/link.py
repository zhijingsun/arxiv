import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import re
from download import download_github_folder 
from link_extract_pdf import read_pdf_from_url, extract_https_links
from dataset_model import dataset_extractor

def extract_urls_from_arxiv(arxiv_url):
    # paper in pdf
    part = arxiv_url.split('/')[3]
    if part == 'pdf':
        pdf_text = read_pdf_from_url(arxiv_url)
        urls = extract_https_links(pdf_text)
        return urls
    
    # paper in html 
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
            prompt = """
                You are an expert dataset URL finder. Find and return dataset URLs from input URL using the following steps: 
                1. Read the content of the input URL, if the page shows a direct 'download' or 'data' button, return the dataset URL either shown or embedded in the button. 
                2. if the input url contains 'GitHub': 
                - Look for the dataset location information in the repository's README file. The information could be the dataset URL embedded or shown. It could also be a directory showing how to find the dataset folder from the main github folder. If so, follow the directory path to trace the location of the dataset URL and return the URL.
                - If the path is not found in the README file, look for the dataset URL in the repository's 'data' or 'dataset' folder or other relevant folders.
                - If such files exist, check inside the file to find whether there are URLs or files containing datasets and return the dataset URL.
                - if there are more than one dataset file, return the URL of the folder that contains all datasets. 
                3.  if the input url contains 'Huggingface': Return the URL to the dataset. 
                4. Other Formats: If the URL contains the word 'json' or 'csv' or other potential formats of datasets at the end of the URL, return the URL directly. 
                5. Return 'Not Found', if no such dataset URL exists.
                Solve data extraction task with interleaving Thought, Action, Observation, Return steps reiterating when necessary. Observation analysis the given information, ie. the content of the page. Thought can help judge whether there is such a url or file containing dataset and determine the next action with the given information, and Action can be three types: 
                (1) Search[path], which searches for the location of file containing datasets in the content of current url and returns the path that shows how to locate the dataset file and direct to the following relevant url. 
                (2) Open[url], which follows the thought to open the next URL or folder. 
                Return would only return the complete final dataset URL, do not give the embedded one. 
                Additionally, look for URLs embedded in specific words or phrases such as 'here', 'this link', or 'download'. Make sure to follow any such links to locate the dataset. 
                """

            datasets = dataset_extractor(prompt=prompt, input_url=url)

            # 剔除包含 "Error: Unable to extract URL from response." 的项
            datasets = [ds for ds in datasets if "Error: Unable to extract URL from response." not in ds]
            
            if len(datasets) > 0:
                st.write(f"Found {len(datasets)} datasets in {url}")
                st.write(datasets)
                all_datasets.extend(datasets)


            # if 'github.com' in url:
            #     datasets = find_datasets_in_github(url)
            #     st.write(f"Found {len(datasets)} datasets in {url}")
            #     st.write(datasets)
            #     all_datasets.extend(datasets)
            # elif 'huggingface.co' in url:
            #     # Add similar logic for handling datasets from HuggingFace
            #     pass
        
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
                        dest_dir = "/Users/zhijingsun/Downloads/" + parts[-1]
                        
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
