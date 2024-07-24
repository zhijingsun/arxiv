import requests
import os
import time
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyDH6BbgjJ5wO5JLuH3sDoJZkOB8P7GMCv0"

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')

#setup
# Define the step function for handling retries
def step(action):
    attempts = 0
    max_attempts = 10
    delay = 2  # Delay between attempts in seconds

    while attempts < max_attempts:
        try:
            # Assuming action is a URL to be requested
            response = requests.get(action)
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()  # Return the JSON content of the response
        except requests.exceptions.Timeout:
            attempts += 1
            time.sleep(delay)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    return {"error": "Max attempts reached"}



def dataset_extractor(prompt=None):
    instruction = """
    Solve data extraction task with interleaving Thought, Action, Observation steps. Thought can help judge whether there is such a url or file containing dataset, and Action can be three types: 
    (1) Think[content], which reads the content and considers whether there is such a url or file containing dataset.
    (2) Search[path], which searches for the location of file containing datasets in the content of current url and 
            returns the path that shows how to locate the dataset file and direct to the following relevant url.
    (3) Finish[url], which returns the specific url or file containing datasets.
    Additionally, look for URLs embedded in specific words or phrases such as 'here', 'this link', or 'download'. Make sure to follow any such links to locate the dataset.
    Here is an example:
    Input URL: 'https://github.com/google-research/google-research/tree/master/youtube_sl_25'
    Thought: this link is from github, access its README file.
    Action: Search: find the line, "You can find the list of video IDs here." and the link is embedded in the word 'here'.
    Thought: access the link
    Action: Finish: return the the url, 'https://console.cloud.google.com/storage/browser/gresearch/youtube-sl-25;tab=objects?prefix=&forceOnObjectsSortingFiltering=false'
    Based on these steps, find and provide the URL for the dataset mentioned.
    """
    
    prompt = instruction + prompt
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        try:
            thought_action = model.generate_content(prompt)
            return thought_action
        except ValueError:
            print('Error')
            n_badcalls += 1
            continue
        except Exception as e:
            if 'User location is not supported' in str(e):
                print('Error: Your location is not supported for API use.')
                break
            print('Unexpected error:', e)
            n_badcalls += 1
            continue
    return None


# Example usage
url = "https://github.com/UKPLab/arxiv2024-lfqa-hallucination/blob/master"  # Replace with the actual URL
prompt = """
Prompt: Find and provide dataset URLs using the following steps:

1. Direct Download: If the page shows a download dataset button, find the URL.
2. GitHub:
    - Look for the dataset location information in the repository's README file.
    - If the path is not found in the README file, look for datasets in the repository's 'data' or 'dataset' folder.
    - If such files exist, check inside the file to find whether there are URLs or files containing datasets.
3. Huggingface: Provide the URL to the dataset.
4. Other Formats: If the link contains the word 'json' or 'csv', provide the URL directly.
"""

result = dataset_extractor(prompt=prompt)
print(result)