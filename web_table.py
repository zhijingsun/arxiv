import json
from typing import Optional
import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/zhijingsun/Desktop/东理/arxiv-analysis-firebase-adminsdk-xrets-eac21e1084.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://arxiv-analysis-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

# Function to fetch data from Firebase
def fetch_data_from_firebase():
    ref = db.reference('/')
    return ref.get()

# Streamlit app configuration
st.set_page_config(
    page_title="Dataset",
    page_icon=":orange_heart:",
)
st.title("AI Research Workflow")
# st.markdown("##### :orange_heart: built by [phidata](https://github.com/phidatahq/phidata)")

# Fetch data from Firebase
data = fetch_data_from_firebase()

# Print data to understand its structure
print(data)

# Flatten the nested JSON structure
flattened_data = []
for category, category_data in data.items():
    if isinstance(category_data, str):
        print(f"Category data for category '{category}' is a string: {category_data}")
        continue
    category_data_list = []
    for url, dataset in category_data.items():
        dataset['Category'] = category
        category_data_list.append(dataset)
    flattened_data.append((category, category_data_list))

# Add a search box
search_query = st.text_input("Search Datasets")

# Filter data based on search query
# 可搜索title和domain
filtered_flattened_data = []
for category, category_data_list in flattened_data:
    filtered_category_data_list = [
    dataset for dataset in category_data_list 
        if ('Title' in dataset and 'Description' in dataset and 'OriginalText' in dataset) and
            (search_query.lower() in dataset['Title'].lower() or
            search_query.lower() in dataset['Description'].lower() or 
            search_query.lower() in dataset['OriginalText'].lower() 
            )
    ]

    if filtered_category_data_list:
        filtered_flattened_data.append((category, filtered_category_data_list))

# Display filtered data
for category, category_data_list in filtered_flattened_data:
    with st.expander(f"{category} Datasets"):
        for dataset in category_data_list:
            st.markdown(f"### {dataset['Title']}")
            st.markdown(f"**Description:** {dataset['Description']}")
            # st.markdown(f"**Domain:** {dataset['Domain']}")
            # st.markdown(f"**Language:** {dataset['Language']}")
            # st.markdown(f"**Size:** {dataset['Size']}")
            # st.markdown(f"**Example Problem:** {dataset['Example']['Problem']}")
            # st.markdown(f"**Example Solution:** {dataset['Example']['Solution']}")
            # st.markdown(f"**URL:** [{dataset['URL']}]({dataset['URL']})")
            st.markdown(f"**context:** {dataset['OriginalText']}")
