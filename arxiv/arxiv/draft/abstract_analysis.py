import logging
import json
import os
from phi.assistant import Assistant
from phi.llm.ollama import Ollama
from phi.llm.groq import Groq
from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.groq import Groq
from textwrap import dedent
from pydantic import BaseModel

# Configuration for logging
# logging.basicConfig(level=logging.DEBUG)


model = "llama3-70b-8192"

class DatasetDescription(BaseModel):
    description: str
    url: str

def analyse_content(pdf_data, text: str, url):
    try:
        # Analyze the README content with JSON output prompt
        assistant = Assistant(
            # llm=Ollama(model="phi3:latest"),
            # llm=Ollama(model="llama3"),
            llm=Groq(model=model),
            instructions=[
                # "You will be provided an arxiv paper with pdf format.",
                # "You will analyze this document and write a report."
                "Your description should follow the format provided below.",
                dedent("""
                    <report_format>
                    {   
                        "Title": "Dataset Title",
                        "Description": "Derive a succinct description about the dataset",
                    }
                    </report_format>
                """)
                ],
            markdown=True,
            debug_mode=True
            )

        json_output_prompt = assistant.get_json_output_prompt()
        full_prompt = f"{json_output_prompt}\n\nAnalyze the following content and extract the required fields:\n\n{text}"
        response = assistant.get_response(full_prompt)
        
        # Parse the response
        response_dict = json.loads(response)
        
        # Add the original text to the response dictionary
        response_dict["OriginalText"] = text
        response_dict["PaperUrl"] = url

        # Convert to JSON format
        json_result = json.dumps(response_dict, indent=4)
        print("Final JSON result:", json_result)

        return json_result

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

