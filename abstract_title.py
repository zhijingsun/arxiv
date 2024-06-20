import requests
import logging
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import json
from link_extract import extract_link_context, extract_https_links

def is_valid_pdf(file_path: str) -> bool:
    """Checks if a file is a valid PDF."""
    try:
        PdfReader(file_path)
        return True
    except PdfReadError:
        return False

def extract_abstract_title_from_pdf(file_path: str) -> str:
    """Extracts the abstract from the PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            
            # Extract text from the first two pages
            for i in range(2):
                if i < len(reader.pages):
                    page = reader.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text.lower()
                        
            if not text:
                return "No text found on the first two pages.", ""

            # Locate and extract the abstract
            introduction_start = text.find("introduction")

            if introduction_start != -1:
                text_before_introduction = text[:introduction_start].strip()
                return text_before_introduction
            else:
                return "Abstract or Introduction section not found on the first page."
    except Exception as e:
        logging.error(f"Failed to read or extract abstract from PDF: {file_path} - {e}")
        raise








