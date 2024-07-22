import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_file(url, dest_folder, file_name):
    try:
        file_path = os.path.join(dest_folder, file_name)
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_path}")
    except requests.RequestException as e:
        print(f"Failed to download {file_name} from {url}: {e}")

def download_github_folder(repo_url, folder_path, dest_dir):
    base_url = f"https://github.com/{repo_url}/tree/main/{folder_path}"
    response = requests.get(base_url)
    response.raise_for_status()  # Check for request errors
    soup = BeautifulSoup(response.text, 'html.parser')
    
    all_files = []
    sub_folders = []
    
    for link in soup.find_all('a'):
        file_link = link.get('href')
        if file_link:
            if file_link.startswith(f"/{repo_url}/blob/main/{folder_path}"):
                # Get the full URL for raw file
                file_url = urljoin('https://raw.githubusercontent.com/', file_link.replace('/blob', ''))
                # Extract file name from the URL
                file_name = os.path.basename(file_link)
                # Track files to download
                all_files.append((file_url, file_name))
            elif file_link.startswith(f"/{repo_url}/tree/main/{folder_path}"):
                sub_folder_path = file_link.split(f"/{repo_url}/tree/main/")[-1]
                sub_dest_dir = os.path.join(dest_dir, sub_folder_path)
                sub_folders.append((sub_folder_path, sub_dest_dir))
    
    print(f"dest_dir: {dest_dir}")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Download files after subdirectories are handled
    for file_url, file_name in all_files:
        print(f"Preparing to download: {file_url}")
        download_file(file_url, dest_dir, file_name)

    # Recursively download subdirectories first
    for sub_folder_path, sub_dest_dir in sub_folders:
        print(f"Downloading subfolder: {sub_folder_path} into {sub_dest_dir}")
        download_github_folder(repo_url, sub_folder_path, sub_dest_dir)    

if __name__ == "__main__":
    repo_url = "bitMyron/sa-swin"
    folder_path = "losses"
    dest_dir = "sa-swin-datasets"  # 想要下载文件到的本地目录
    
    download_github_folder(repo_url, folder_path, dest_dir)
    
    # 打印下载目录的绝对路径
    print(f"Files downloaded to: {os.path.abspath(dest_dir)}")