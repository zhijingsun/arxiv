import logging
import json
from link_extract import read_pdf, read_pdf_first_page
from phi.assistant import Assistant
from phi.llm.ollama import Ollama
from phi.llm.groq import Groq

TAVILY_API_KEY = "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"
GROQ_API_KEY = "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"

model = "llama3-70b-8192"


pdf_path = "/Users/zhijingsun/Desktop/arxiv_paper_update/5_Jun_2024/IR/2406.01603.pdf"
pdf_content = read_pdf(pdf_path)
pdf_content_first = read_pdf_first_page(pdf_path)


assistant = Assistant(
    # llm=Ollama(model="phi3:latest"),
    # llm=Ollama(model="llama3"),
    llm=Groq(model=model),
    instructions=[
    # "You will be provided an arxiv paper with pdf format.",
    # "You will analyze this document and write a report."
    f"""analyze this document, "{pdf_content_first}", in the format:

    - **Overview** Brief introduction of the abstract.
    """
    ],
    markdown=True,
)


assistant.get_response(f"analyze this document with the given format in instruction: {pdf_content_first}")

# try:
#     # 调用assistant进行分析
#     response = assistant.print_response(f"analyze this document with the given format: {pdf_content}")

#     # 转换为JSON格式字符串
#     json_result = json.dumps(result, indent=4)
#     print(json_result)
# except Exception as e:
#     logging.error(f"An error occurred: {str(e)}")

# print(type(pdf_id)) 
# print(type(response)) 
