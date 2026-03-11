import feedparser

CERN_RSS = "https://home.cern/news/rss"

def fetch_cern_news(query):
    results = []
    feed = feedparser.parse(CERN_RSS)

    aliases = {
        query.lower(),
        "lhc",
        "large hadron collider"
    }

    for entry in feed.entries[:15]:
        text = f"{entry.title}. {entry.summary}".lower()
        if any(a in text for a in aliases):
            results.append({
                "text": entry.title + ". " + entry.summary,
                "source": "CERN",
                "title": entry.title,
                "year": entry.published[:4] if hasattr(entry, "published") else None,
                "url": entry.link,
                "authority": 0.95
            })

    return results