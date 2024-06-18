# IMPORT LIBRARIES
import json
import requests
import sys


base_url = "https://arxiv-analysis-default-rtdb.asia-southeast1.firebasedatabase.app/"


def add_dataset_info(category, url, info_json):
    # INPUT : pdf_id and pdf_json from command line
    # RETURN : status code after pyhton REST call to add paper [response.status_code]
    # EXPECTED RETURN : 200
    DATABASE_URL = f"{base_url}{category}.json"
    info_data = json.loads(info_json)
    response = requests.patch(DATABASE_URL,json={str(url):info_data})
    print(response.status_code)

