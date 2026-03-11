import requests
from bs4 import BeautifulSoup

INSTITUTION_SITES = [
    "site:cdc.gov",
    "site:nih.gov",
    "site:who.int",
    "site:nasa.gov",
    "site:noaa.gov",
    "site:edu"
]

def search_institutions(query, max_results=3):
    """
    Lightweight institutional search using DuckDuckGo HTML
    (no API key, fast, reliable)
    """
    results = []

    for site in INSTITUTION_SITES:
        q = f"{query} {site}"
        url = "https://duckduckgo.com/html/"

        try:
            r = requests.post(
                url,
                data={"q": q},
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
        except Exception:
            continue

        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.select(".result__a")[:max_results]:
            link = a.get("href")
            title = a.text.strip()

            results.append({
                "text": title,
                "source": "Institution",
                "title": title,
                "url": link,
                "year": None,
                "authority": 0.95
            })

    return results
