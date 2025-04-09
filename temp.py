import requests
from bs4 import BeautifulSoup

try:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get("https://getcohesiveai.com/scraper", headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    s = " ".join(text.split())
    print(s)
except Exception as e:
    print(f"Error scraping: {e}")
