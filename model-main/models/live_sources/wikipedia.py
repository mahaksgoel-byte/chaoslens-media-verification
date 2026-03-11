import requests

API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "MisinfoDetector/1.0 (academic project)"
}

def fetch_wikipedia_page(entity_label):
    """
    Universal Wikipedia entity resolution:
    search → pageid → extract
    """

    # 1️⃣ search for the entity
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": entity_label,
        "format": "json",
        "srlimit": 1
    }

    try:
        r = requests.get(API_URL, params=search_params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return []

    pageid = search_results[0]["pageid"]
    title = search_results[0]["title"]

    # 2️⃣ fetch the exact page by pageid
    page_params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "pageids": pageid,
        "format": "json"
    }

    try:
        r = requests.get(API_URL, params=page_params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        page_data = r.json()
    except Exception:
        return []

    pages = page_data.get("query", {}).get("pages", {})
    page = pages.get(str(pageid), {})
    text = page.get("extract")

    if not text:
        return []

    return [{
        "text": text[:1200],
        "source": "Wikipedia",
        "title": title,
        "year": None,
        "authors": [],
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "authority": 0.9
    }]
