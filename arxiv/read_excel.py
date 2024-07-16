
import pandas as pd
import json
import logging

def excel_sheet_to_json(file_path: str, sheet_index: int = 1, link_column: str = 'url', label_column: str = 'label') -> str:
    """
    Reads a specific sheet from an Excel file and extracts specified columns,
    then converts the data to JSON format.

    Parameters:
    - file_path: str - Path to the Excel file.
    - sheet_index: int - Index of the sheet to read (0-based).
    - link_column: str - The name of the column containing the paper links.
    - label_column: str - The name of the column containing the labels.

    Returns:
    - str - JSON formatted string.
    """
    try:
        # Read the specific sheet from the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_index)

        # Extract the specified columns
        data = df[[link_column, label_column]]

        # Convert the DataFrame to a list of dictionaries
        data_list = data.to_dict(orient='records')
        return data_list
    except Exception as e:
        logging.error(f"Failed to read Excel file or extract columns: {file_path} - {e}")
        raise

# Example usage
#file_path = '/Users/guowj/Desktop/llm/phidata/arxiv_v1/datasetcollection.xlsx'
#json_output = excel_sheet_to_json(file_path, sheet_index=1)  # Reading the second sheet (0-based index)




