import logging
import requests


# # 配置日志记录
# logging.basicConfig(level=logging.DEBUG)

#url_link = "https://github.com/Paitesanshi/LLM-Agent-Survey" #random link
url_link = "https://github.com/openai/grade-school-math"
    

def encode_url(url: str) -> str:
    """Encodes a URL to a Firebase-compatible key.
    
    Args:
        url (str): The URL to encode.
    
    Returns:
        str: The encoded URL.
    """
    # Replace special characters with acceptable characters
    return url.replace('.', '_').replace('$', '_').replace('#', '_').replace('[', '_').replace(']', '_').replace('/', '_')

def is_invalid_link(url: str) -> bool:
    """Check if the link is invalid, shows 'code is coming soon', or returns a 404 error.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the link is invalid, shows 'code is coming soon', or returns a 404 error, False otherwise.
    """
    try:
        response = requests.get(url)
        # Check if the URL returns a 404 error
        if response.status_code == 404:
            return True
        content = response.text
        # Check if the content is empty or shows 'code is coming soon'
        if not content or 'code is coming soon' in content.lower():
            return True
    except requests.RequestException:
        # If there's an issue with the request itself, we can assume the link is invalid
        return True
    return False


def process_content(pdf_data, url_link: str):
    # print(url_link)
    try:
        # Fetch the content
        response = requests.get(url_link)
        
        if response.status_code != 200 or is_invalid_link(response.text):
            print("Invalid")
            #add data collection?
            return

        # Add to database
        encoded_url = encode_url(url_link)
        return encoded_url

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

# 示例调用
#process_content(,"https://huggingface.co/facebook/wav2vec2-base")