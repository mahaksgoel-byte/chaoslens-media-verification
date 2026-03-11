import requests
import time

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_semantic_scholar(query, limit=5):
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,authors,url"
    }

    try:
        r = requests.get(API_URL, params=params, timeout=10)

        # handle rate limiting gracefully
        if r.status_code == 429:
            time.sleep(2)
            return []

        r.raise_for_status()
        data = r.json()

    except Exception:
        return []

    results = []
    for paper in data.get("data", []):
        if not paper.get("abstract"):
            continue

        results.append({
            "text": paper["abstract"],
            "source": "Semantic Scholar",
            "title": paper.get("title"),
            "year": paper.get("year"),
            "authors": [a["name"] for a in paper.get("authors", [])],
            "url": paper.get("url"),
            "authority": 0.95
        })

    return results
