import requests

API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def search_pubmed(query, limit=5):
    params = {
        "query": query,
        "format": "json",
        "pageSize": limit,
        "resultType": "core"
    }

    r = requests.get(API_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("resultList", {}).get("result", []):
        abstract = item.get("abstractText")
        year = item.get("pubYear")

        if not abstract or not year:
            continue

        results.append({
            "text": abstract,
            "source": "PubMed",
            "title": item.get("title"),
            "year": int(year),
            "authors": item.get("authorString", "").split(", "),
            "url": f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}",
            "authority": 0.95
        })

    return results
