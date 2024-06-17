import re

def extract_https_links(text: str) -> list:
    """从文本中提取所有HTTPS链接，并去除'arXiv:'部分"""
    # 提取所有的 HTTPS 链接
    all_links = re.findall(r'https://\S+', text)
    # 去除每个链接中包含的 'arXiv:' 部分
    cleaned_links = [re.sub(r'arXiv:\S+', '', link).strip() for link in all_links]
    return cleaned_links

# 示例文本
text = "Our code and MultiLJP dataset are available at https://github.com/CURRENTF/HRN. For more details, visit https://github.com/misonsky/AmdgarXiv:2406.09881v1."

# 提取链接
links = extract_https_links(text)
print(links)
