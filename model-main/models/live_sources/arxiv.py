import feedparser
import urllib.parse

BASE_URL = "http://export.arxiv.org/api/query"

def search_arxiv(query, max_results=5):
    # sanitize + encode query
    safe_query = urllib.parse.quote(query)

    search_query = f"search_query=all:{safe_query}&start=0&max_results={max_results}"
    feed = feedparser.parse(f"{BASE_URL}?{search_query}")

    results = []

    for entry in feed.entries:
        results.append({
            "text": entry.summary,
            "source": "arXiv",
            "title": entry.title,
            "year": entry.published[:4],
            "authors": [a.name for a in entry.authors],
            "url": entry.link,
            "authority": 0.7
        })

    return results
