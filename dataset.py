import requests
import logging
import time
import json
from extract_abstract import is_valid_pdf, extract_abstract_from_pdf
from read_excel import excel_sheet_to_json
#from local.abstract_context_extract import extract_abstract_context_from_pdf
#from abstract_title import extract_abstract_title_from_pdf
from extract_title import extract_title_from_pdf

def read_pdf_abstract_from_url(url: str) -> dict:
    """Reads an online PDF file and extracts the abstract and title."""
    try:
        # Fetch the PDF from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Save the PDF to a temporary file
        with open('/tmp/temp.pdf', 'wb') as f:
            f.write(response.content)

        # Check if the PDF is valid
        if not is_valid_pdf('/tmp/temp.pdf'):
            logging.warning(f"Invalid PDF file for URL: {url}")
            return None, None

        # Extract the abstract and context from the saved PDF
        #text_before_introduction = extract_abstract_title_from_pdf('/tmp/temp.pdf')
        #return text_before_introduction

        title = extract_title_from_pdf('/tmp/temp.pdf')
        print(title)
        return title
        # abstract = extract_abstract_from_pdf('/tmp/temp.pdf')
        # return abstract
        

    except requests.RequestException as e:
        logging.error(f"Failed to fetch PDF from URL: {url} - {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return None, None

def output_from_excel(file_path: str):
    excel_info = excel_sheet_to_json(file_path, sheet_index=1)
    output = []
    for row in excel_info:
        if not isinstance(row['paper link'], str):
            continue
        label = row['label']
        url = row['paper link']
        text_before_introduction = read_pdf_abstract_from_url(row['paper link'])
        if text_before_introduction is None:
            continue
        one_set = {"url": url, "label": label, "text_before_introduction": text_before_introduction}
        output.append(one_set)
        time.sleep(2)
    return output 

if __name__ == "__main__":
    file_path = '~/desktop/datasetcollection.xlsx'
    result = output_from_excel(file_path)
    # Write the results to a JSON file
    output_file = 'dataset.json'
    with open(output_file, 'w') as outfile:
        json.dump(result, outfile, indent=4)

    print(f"All abstracts extracted and saved to {output_file}")
