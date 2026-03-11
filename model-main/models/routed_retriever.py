from models.live_sources.wikipedia import fetch_wikipedia_page
from models.live_sources.pubmed import search_pubmed
from models.live_sources.arxiv import search_arxiv
from models.live_sources.semantic_scholar import search_semantic_scholar
from models.live_sources.institutions import search_institutions
from sentence_transformers import SentenceTransformer, util


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def detect_claim_intent(claim):
    c = claim.lower()

    if any(k in c for k in ["born", "developed", "invented", "discovered"]):
        return "biography"

    if any(k in c for k in ["shut down", "failure", "destroyed", "confirmed", "reported"]):
        return "event"

    if any(k in c for k in ["causes", "leads to", "results in", "associated with"]):
        return "causal"

    return "general"


def _entity_consistent(text, entity_label):
    # ONLY for non-Wikipedia sources
    if not text or not entity_label:
        return False
    return entity_label.lower() in text.lower()


def fetch_routed_evidence(claim, entity=None, intent=None):
    evidence = []

    # ==================================
    # ENTITY PRESENT → ALWAYS WIKIPEDIA
    # ==================================
    if entity and entity.get("label"):
        label = entity["label"]
        wiki_evidence = fetch_wikipedia_page(label)
        if wiki_evidence:
            evidence.extend(wiki_evidence)

        # ENTITY FACTS STOP HERE
        if intent == "ENTITY_FACT":
            return evidence

    # ==================================
    # SCIENTIFIC / MEDICAL
    # ==================================
    if intent in {"SCIENTIFIC", "MEDICAL"}:
        evidence += search_semantic_scholar(claim)
        evidence += search_arxiv(claim)
        evidence += search_pubmed(claim)
        return evidence

    # ==================================
    # EVENT
    # ==================================
    if intent == "EVENT":
        return search_institutions(claim)

    return evidence
