import requests
from bs4 import BeautifulSoup
import json

# Base URL of the Arxiv cs.SC recent submissions
base_url = "https://arxiv.org/list/cs.SC/recent"

# Make a request to fetch the HTML content of the page
response = requests.get(base_url)
html_content = response.content

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the section with the latest submissions
recent_submissions = soup.find_all('dl')[0]

# Extract titles, abstracts, and URLs from the latest submissions
papers = []
for dt, dd in zip(recent_submissions.find_all('dt'), recent_submissions.find_all('dd')):
    try:
        paper_url = "https://arxiv.org" + dt.find('a', title='Abstract')['href']
        paper_response = requests.get(paper_url)
        paper_soup = BeautifulSoup(paper_response.content, 'html.parser')
        
        title = paper_soup.find('h1', class_='title mathjax').text.strip().replace('Title:', '')
        abstract = paper_soup.find('blockquote', class_='abstract mathjax').text.strip().replace('Abstract:', '')
        
        pdf_url = "https://arxiv.org" + dt.find('a', title='Download PDF')['href']
        
        papers.append({
            'url': pdf_url,
            'title': title,
            'abstract': abstract
        })
    except Exception as e:
        print(f"Error fetching details for a paper: {e}")

# Save the results to a JSON file
with open('latest_papers.json', 'w') as json_file:
    json.dump(papers, json_file, indent=4)

# Print the results
for paper in papers:
    print(json.dumps(paper, indent=4))
