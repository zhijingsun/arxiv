import requests
import logging
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import json

#logging.basicConfig(level=logging.INFO)


# def read_pdf_abstract_from_url(url: str) -> str:
#     """Reads an online PDF file and extracts the abstract."""
#     try:
#         # Fetch the PDF from the URL
#         response = requests.get(url)
#         response.raise_for_status()

#         # Save the PDF to a temporary file
#         with open('/tmp/temp.pdf', 'wb') as f:
#             f.write(response.content)

#         # Check if the PDF is valid
#         if not is_valid_pdf('/tmp/temp.pdf'):
#             return "Invalid PDF file."

#         # Extract the abstract from the saved PDF
#         abstract = extract_abstract_from_pdf('/tmp/temp.pdf')
#         return json.dumps({"url": url, "abstract": abstract})
#     except requests.RequestException as e:
#         logging.error(f"Failed to fetch PDF from URL: {url} - {e}")
#         raise

def is_valid_pdf(file_path: str) -> bool:
    """Checks if a file is a valid PDF."""
    try:
        PdfReader(file_path)
        return True
    except PdfReadError:
        return False

def extract_abstract_from_pdf(file_path: str) -> str:
    """Extracts the abstract from the PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = reader.pages[0].extract_text()
            text += reader.pages[1].extract_text()

            # Convert text to lowercase for case-insensitive search
            text = text.lower()
            # Locate and extract the abstract
            abstract_start = text.find("abstract")
            introduction_start = text.find("introduction")
            
            if abstract_start != -1 and introduction_start != -1:
                abstract = text[abstract_start + len("abstract"):introduction_start].strip()
                abstract = abstract.replace('\n', ' ')
                return abstract
            else:
                return "Abstract or Introduction section not found on the first page."
    except Exception as e:
        logging.error(f"Failed to read or extract abstract from PDF: {file_path} - {e}")
        raise





# # Example usage
# url = 'https://arxiv.org/pdf/2406.08979'
# abstract = read_pdf_abstract_from_url(url)
# print(abstract)



