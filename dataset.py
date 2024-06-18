import requests
import logging
import time
import json
from extract_abstract import is_valid_pdf, extract_abstract_from_pdf
from crawl_arxiv_url import get_pdf_urls

def read_pdf_abstract_from_url(url: str) -> dict:
    """Reads an online PDF file and extracts the abstract."""
    try:
        # Fetch the PDF from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Save the PDF to a temporary file
        with open('/tmp/temp.pdf', 'wb') as f:
            f.write(response.content)

        # Check if the PDF is valid
        if not is_valid_pdf('/tmp/temp.pdf'):
            return {"url": url, "abstract": "Invalid PDF file."}

        # Extract the abstract from the saved PDF
        abstract = extract_abstract_from_pdf('/tmp/temp.pdf')
        print(abstract)
        return {"url": url, "abstract": abstract}

    except requests.RequestException as e:
        logging.error(f"Failed to fetch PDF from URL: {url} - {e}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    pdf_urls = get_pdf_urls()
    abstracts = []

    for pdf_data in pdf_urls:
        abstract_info = read_pdf_abstract_from_url(str(pdf_data['pdf_link']))
        abstracts.append(abstract_info)
        time.sleep(2)  # Delay before the next iteration

    # Write all abstracts to a JSON file
    output_file = 'all_abstracts.json'
    with open(output_file, 'w') as outfile:
        json.dump(abstracts, outfile, indent=4)

    print(f"All abstracts extracted and saved to {output_file}")
