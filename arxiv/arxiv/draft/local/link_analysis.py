import logging
import requests
import re
import json
import os
from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.groq import Groq
from textwrap import dedent
from arxiv.draft.database import add_dataset_info
from pydantic import BaseModel

# # 配置日志记录
# logging.basicConfig(level=logging.DEBUG)

TAVILY_API_KEY="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"
GROQ_API_KEY="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"

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model = "llama3-70b-8192"

#url_link = "https://github.com/Paitesanshi/LLM-Agent-Survey" #random link
url_link = "https://github.com/openai/grade-school-math"
    
def extract_first_paragraph(text: str) -> str:
    """Extract the first paragraph from the text.

    Args:
        text (str): Text content to extract the first paragraph from.

    Returns:
        str: The first paragraph of the text.
    """
    paragraphs = text.split('\n\n')
    return paragraphs[0].strip() if paragraphs else ''

def encode_url(url: str) -> str:
    """Encodes a URL to a Firebase-compatible key.
    
    Args:
        url (str): The URL to encode.
    
    Returns:
        str: The encoded URL.
    """
    # Replace special characters with acceptable characters
    return url.replace('.', '_').replace('$', '_').replace('#', '_').replace('[', '_').replace(']', '_').replace('/', '_')

class DatasetDescription(BaseModel):
    domain: str
    description: str
    language: str
    url: str
    size: str
    example: dict

def is_invalid_link(content: str) -> bool:
    """Check if the link content is invalid or the page shows 'code is coming soon'.

    Args:
        content (str): The content of the page.

    Returns:
        bool: True if the link is invalid or shows 'code is coming soon', False otherwise.
    """
    if not content or 'code is coming soon' in content.lower():
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
        
        # Fetch the content
        first_paragraph = extract_first_paragraph(response.text)

        # Analyze the README content with JSON output prompt
        assistant = Assistant(
            llm=Groq(model=model),
            instructions=[
                "Your description should follow the format provided below.",
                dedent("""
                    <report_format>
                    {
                        "Title": "Dataset Title",
                        "Domain": "Name of the domain of the dataset",
                        "Description": "A succinct description of the dataset",
                        "Language": "The language used in the dataset",
                        "Size": "The size of the dataset, such as the number of records, entries, or data points",
                        "Example": {
                            "Problem": "An example problem from the dataset",
                            "Solution": "The solution to the example problem"
                        },
                        "URL": "URL of the dataset"
                    }
                    </report_format>
                """)
            ],
            markdown=True,
            debug_mode=True
        )

        json_output_prompt = assistant.get_json_output_prompt()
        full_prompt = f"{json_output_prompt}\n\nAnalyze the following content and extract the required fields:\n\n{first_paragraph}"
        response = assistant.get_response(full_prompt)
        
        # Parse the response
        response_dict = json.loads(response)

        # Split the example into problem and solution    
        #response_dict['Example'] = response_dict.get('example', '')

        # Add the URL to the result dictionary
        response_dict['URL'] = url_link

        # Convert to JSON format
        json_result = json.dumps(response_dict, indent=4)
        print("Final JSON result:", json_result)
        print(type(json_result))

        # Add to database
        encoded_url = encode_url(url_link)
        add_dataset_info(pdf_data['category'], encoded_url, json_result)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

# 示例调用
#process_github_readme("https://github.com/Paitesanshi/LLM-Agent-Survey")
# process_content("https://huggingface.co/facebook/wav2vec2-base")