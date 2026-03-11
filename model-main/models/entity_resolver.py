import re
import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

GENERIC_TERMS = {
    "some", "many", "engineers", "scientists", "researchers",
    "officials", "people", "experts", "they", "we", "he", "she"
}

FACILITY_KEYWORDS = {
    "collider", "accelerator", "reactor", "telescope",
    "laboratory", "facility", "observatory", "station"
}


class EntityResolver:
    def extract_candidate(self, claim):
        matches = re.findall(
            r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,4}", claim
        )

        for m in matches:
            token = m.lower().strip()

            if token in GENERIC_TERMS:
                continue

            words = token.split()

            # Reject single generic capitalized words
            if len(words) == 1 and words[0] in GENERIC_TERMS:
                continue

            return m

        return None

    def is_valid_entity(self, candidate):
        words = candidate.lower().split()

        # Person names (Albert Einstein)
        if len(words) >= 2:
            return True

        # Facilities must contain a keyword
        if any(k in words for k in FACILITY_KEYWORDS):
            return True

        return False

    def resolve(self, claim):
        candidate = self.extract_candidate(claim)
        if not candidate:
            return None

        if not self.is_valid_entity(candidate):
            return None

        params = {
            "action": "wbsearchentities",
            "search": candidate,
            "language": "en",
            "format": "json",
            "limit": 1
        }

        try:
            r = requests.get(WIKIDATA_API, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return {"label": candidate, "qid": None}

        results = data.get("search", [])
        if not results:
            return {"label": candidate, "qid": None}

        top = results[0]
        return {
            "label": top["label"],
            "qid": top.get("id"),
        }
