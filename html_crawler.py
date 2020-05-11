from bs4 import BeautifulSoup # BeautifulSoup is in bs4 package
import requests

URL = 'https://www.cnn.com/2015/11/28/opinions/sutter-cop21-paris-preview-two-degrees/index.html'
content = requests.get(URL)

soup = BeautifulSoup(content.text, 'html.parser')

row = soup.find('tr') # Extract and return first occurrence of tr
print(row)            # Print row with HTML formatting
print("=========Text Result==========")
print(row.get_text()) # Print row as text

