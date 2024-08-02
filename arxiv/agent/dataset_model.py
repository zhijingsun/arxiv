import requests
import time
import os
from groq import Groq
import re

# 配置API密钥
GROQ_API_KEY = "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"

# 初始化Groq客户端
client = Groq(
    api_key=GROQ_API_KEY
)

# 定义处理重试的步骤函数
def step(action):
    attempts = 0
    max_attempts = 10
    delay = 2  # 尝试之间的延迟时间（秒）

    while attempts < max_attempts:
        try:
            # 假设 action 是要请求的 URL
            response = requests.get(action)
            response.raise_for_status()  # 对不好的响应引发错误
            return response.json()  # 返回响应的 JSON 内容
        except requests.exceptions.Timeout:
            attempts += 1
            time.sleep(delay)  # 重试前等待
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    return {"error": "Max attempts reached"}

def dataset_extractor(prompt, input_url):
    example = """
    Here is an example that we can find the dataset path from README file: 
    Input URL: 'https://github.com/google-research/google-research/tree/master/youtube_sl_25' 
    Thought: this link is from github, access its README file. 
    Action: Search: find the line, "You can find the list of video IDs here." and the link is embedded in the word 'here'. 
    Thought: access the link Action: 
    Finish: return the url, 'https://console.cloud.google.com/storage/browser/gresearch/youtube-sl-25;tab=objects?prefix=&forceOnObjectsSortingFiltering=false'

    Here is an example that we can't find the dataset path from README file: 
    Input URL: 'https://github.com/bitMyron/sa-swin' 
    Thought: this link is from github, access its README file. 
    Action: Search: the dataset path is not found in the README file, look for the dataset URL in the repository's 'data' or 'dataset' folder or other relevant folders. 
    Thought: access the link Action: 
    Finish: return the url, 'https://github.com/bitMyron/sa-swin/tree/main/datasets'

    Additional details:
    If the dataset link contains amazonaws, follow this format 'https://bubble-ml-simulations.s3.us-east-2.amazonaws.com/datasetname.tar.gz' 
    If the dataset link requires the user to fill in certain personal information, output 'please fill in the form to download the dataset from this {url}'

    Based on these steps, find and provide the dataset URL for this input URL: {input_url}.
    """
    
    prompt = prompt + "\n" + example + "\nInput URL: " + input_url + "\n"
    n_calls, n_badcalls = 0, 0
    
    while n_calls < 7:
        n_calls += 1
        try:
            print("Attempt:", n_calls)
            # 调用 Groq 的 chat 资源接口生成内容
            llm = client.chat.completions.create(
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI Assistant. You provide information and assistance based on specific instructions. Your responses should be clear and focused on following the steps provided."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-70b-8192",
            )
            response = llm.choices[0].message.content
            print(response)

            # 提取最后一段文本
            last_paragraph = response.strip().split('\n')[-1]
            # 使用正则表达式提取最后一段中的 URL
            url_match = re.findall(r'https?://[^\s`]+(?<![\]\'`.,!?])', last_paragraph)
            url_match = list(set(url_match))  # 去除重复项
            url_match = [url for url in url_match if url != input_url]  # 剔除与input_url相同的项
            
            if url_match:
                return url_match
            else:
                return ["Error: Unable to extract URL from response."]
            
        except ValueError as ve:
            print('ValueError:', ve)
            n_badcalls += 1
        except Exception as e:
            print('Unexpected error:', e)
            n_badcalls += 1

    return "Error: Unable to retrieve dataset URL after multiple attempts."

# Example usage
url = "https://github.com/bitMyron/sa-swin"  # Replace with the actual URL
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

result = dataset_extractor(prompt=prompt, input_url=url)
print(result)
