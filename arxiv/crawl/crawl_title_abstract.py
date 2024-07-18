import requests
from bs4 import BeautifulSoup
import json
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi

# 基础URL
base_url = "https://arxiv.org"

# 要爬取的类别和页面URL
categories = [
    'cs.IR',
    'cs.DB',
    'cs.AI',
    'cs.CL',
    'cs.CV',
    'cs.MA'
]

# 爬取和解析函数
def fetch_papers_from_url(url):
    try:
        # 设置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        response = http.get(url, timeout=10)  # 增加超时设置为10秒钟
        response.raise_for_status()  # 检查请求是否成功

        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        recent_submissions = soup.find_all('dl')[0]

        papers = []
        for dt, dd in zip(recent_submissions.find_all('dt'), recent_submissions.find_all('dd')):
            try:
                paper_url = base_url + dt.find('a', title='Abstract')['href']
                paper_response = http.get(paper_url, timeout=10)  # 增加超时设置为10秒钟
                paper_soup = BeautifulSoup(paper_response.content, 'html.parser')

                title = paper_soup.find('h1', class_='title mathjax').text.strip().replace('Title:', '').strip()
                abstract = paper_soup.find('blockquote', class_='abstract mathjax').text.strip().replace('Abstract:', '').strip()

                pdf_url = base_url + dt.find('a', title='Download PDF')['href']
                html_url = dt.find('a', title='View HTML')['href']

                papers.append({
                    'url': html_url,
                    'title': title,
                    'abstract': abstract
                })
            except Exception as e:
                print(f"Error fetching details for a paper: {e}")

        return papers

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

# 获取每个类别的更新篇数和日期
def get_entries_count_and_date(url):
    try:
        response = requests.get(url, timeout=10)  # 增加超时设置为10秒钟
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

# 收集所有类别的论文并保存到JSON文件
all_papers = []
for category in categories:
    url = f"{base_url}/list/{category}/recent?skip=0&show=5"
    entries_count, date_str = get_entries_count_and_date(url)
    if entries_count is None:
        continue

    # 替换 show 参数为更新篇数
    url = re.sub(r'show=\d+', f'show={entries_count}', url)

    papers = fetch_papers_from_url(url)
    all_papers.extend(papers)

    # 打印类别和更新篇数
    print(f"{category}: {entries_count} papers updated on {date_str}")

# 保存结果到JSON文件
with open('latest_papers.json', 'w') as json_file:
    json.dump(all_papers, json_file, indent=4)

# # 打印结果
# for paper in all_papers:
#     print(json.dumps(paper, indent=4))