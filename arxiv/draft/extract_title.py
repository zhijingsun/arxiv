import re
import logging
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
#from extract_abstract import is_valid_pdf, extract_abstract_from_pdf
#from dataset import read_pdf_abstract_from_url
import requests

def clean_text(text):
    # This regex removes the arXiv ID and date part, e.g., "arxiv:2406.08726v1  [cs.cl]  13 jun 2024"
    clean_pattern = re.compile(r'arxiv:\d{4}\.\d{5}v\d+\s+\[\w+\.\w+\]\s+\d{1,2}\s\w+\s\d{4}', re.IGNORECASE)
    return clean_pattern.sub('', text)

def extract_title(text):
    # This regex captures the title until the third newline character or a recognizable institution keyword
    title_pattern = re.compile(r'^(.*?)(?:\n.*?){3}|\b(university|college|institute)\b', re.DOTALL)
    
    text = clean_text(text)

    # Search for the pattern in the text
    match = title_pattern.search(text)
    
    # If a match is found, return the title
    if match:
        #print(match.group(0))
        # Extract the matched group before the first keyword or newline limit
        title = match.group(0).strip() if match.group(0) else title
        title = title.replace("\n", " ")
        # Ensure to stop before any institution keywords
        for keyword in ['university', 'college', 'institute', "∗", ",", "⋆", "*", "@"]:
            if keyword in title.lower():
                title = title.split(keyword, 1)[0].strip()
        if len(title) > 110:
            title = title[:110]
        return title
    
    # If no match is found, return None
    return None
def is_valid_pdf(file_path: str) -> bool:
    """Checks if a file is a valid PDF."""
    try:
        PdfReader(file_path)
        return True
    except PdfReadError:
        return False

def extract_title_from_pdf(file_path: str) -> str:
    """Extracts the abstract from the PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = reader.pages[0].extract_text().lower()
            title = extract_title(text)
            return title
    except Exception as e:
        logging.error(f"Failed to read or extract abstract from PDF: {file_path} - {e}")
        raise



#print(read_pdf_abstract_from_url("https://arxiv.org/pdf/2406.08920"))

