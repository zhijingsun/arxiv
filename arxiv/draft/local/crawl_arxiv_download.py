import os
import requests
from bs4 import BeautifulSoup
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi

def get_entries_count_and_date(url):
    try:
        response = requests.get(url, verify=certifi.where())
        response.raise_for_status()  # 检查请求是否成功
        soup = BeautifulSoup(response.text, 'html.parser')
        entries_header = soup.find('h3')
        if entries_header and 'entries' in entries_header.text:
            text = entries_header.text.strip()
            count_match = re.search(r'of (\d+) entries', text)
            date_match = re.search(r'(\d+ \w+ \d+)', text)
            entries_count = int(count_match.group(1)) if count_match else None
            date_str = date_match.group(1) if date_match else None
            if entries_count and date_str:
                return entries_count, date_str
            else:
                print("Could not extract entries count or date from text.")
                return None, None
        else:
            print("Entries header not found.")
            return None, None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None, None

def crawl_arxiv_and_save_pdf():
    base_url = 'https://arxiv.org'
    list_url = [
        # f'{base_url}/list/cs.IR/recent?skip=0&show=5',
        # f'{base_url}/list/cs.DB/recent?skip=0&show=5',
        f'{base_url}/list/cs.AI/recent?skip=0&show=5',
        # f'{base_url}/list/cs.CL/recent?skip=0&show=5',
        # f'{base_url}/list/cs.CV/recent?skip=0&show=5',
        # f'{base_url}/list/cs.MA/recent?skip=0&show=5'
    ]

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    save_dir = "/Users/zhijingsun/Desktop/arxiv_paper_update"  # PDF文件保存路径

    for url in list_url:
        entries_count, date_str = get_entries_count_and_date(url)
        print(f"Entries count: {entries_count}, Date: {date_str}")
        if entries_count is None:
            print(f"Skipping URL due to error: {url}")
            continue
        
        modified_url = re.findall(r'\D+\d\D+', url)[0] + str(entries_count) # 修改https://arxiv.org/list/cs.IR/recent?skip=0&show=5最后的show=
        try:
            response = http.get(modified_url, verify=certifi.where())
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 以日期创建外部文件夹
            date_dir = date_str.replace(" ", "_")
            date_dir_path = os.path.join(save_dir, date_dir)
            if not os.path.exists(date_dir_path):
                os.makedirs(date_dir_path)

            file_name = re.findall(r'\b[A-Z]+\b', url)
            dir_name = os.path.join(date_dir_path, file_name[0])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            papers = soup.find('dl', id='articles')
            if papers is None:
                print(f"No papers found for URL: {modified_url}")
                continue

            items = papers.find_all(['dt', 'dd'])
            for i in range(0, len(items), 2):
                dt = items[i]
                dd = items[i + 1]
                title_tag = dd.find('div', class_='list-title')
                title = title_tag.text.strip().replace('Title:', '').strip() if title_tag else f"paper_{i//2+1}"

                link_tag = dt.find('a', title='Download PDF')
                if link_tag:
                    pdf_link = link_tag['href']
                    if not pdf_link.startswith('http'):
                        pdf_link = base_url + pdf_link

                    pdf_response = http.get(pdf_link, verify=certifi.where())
                    if pdf_response.status_code == 200:
                        paper_id = pdf_link.split('/')[-1]
                        paper_name = f"{paper_id}.pdf"
                        file_path = os.path.join(dir_name, paper_name)
                        with open(file_path, 'wb') as file:
                            file.write(pdf_response.content)
                        print(f"Saved {title} as {paper_name}")
                    else:
                        print(f"Failed to download PDF for {title}")
                else:
                    print(f"No PDF link for {title}")
        except requests.RequestException as e:
            print(f"Request failed for URL {modified_url}: {e}")

