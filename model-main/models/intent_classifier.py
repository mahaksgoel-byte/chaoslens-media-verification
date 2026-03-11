from sentence_transformers import SentenceTransformer, util


class ClaimIntentClassifier:
    """
    Embedding-based intent classifier.
    NO regex rules.
    NO keyword if-else.
    Fully semantic.
    """

    INTENTS = {

        # =========================
        # BIOGRAPHY / ENTITY FACTS
        # =========================
        "ENTITY_FACT": [
            "was born in",
            "was a famous scientist",
            "was a football player",
            "is a singer",
            "is known for",
            "won the Nobel Prize",
            "received an award",
            "died in",
            "is still alive",
            "was not a scientist",
            "never worked as",
            "his profession was",
            "her career was"
        ],

        # =========================
        # EVENTS / INCIDENTS
        # =========================
        "EVENT": [
            "was shut down",
            "was temporarily closed",
            "suffered a failure",
            "incident occurred",
            "was destroyed",
            "officials confirmed",
            "investigation found",
            "engineers reported",
            "repairs were completed",
            "accident happened",
            "operations resumed"
        ],

        # =========================
        # SCIENTIFIC CLAIMS
        # =========================
        "SCIENTIFIC": [
            "developed a theory",
            "proposed the theory of relativity",
            "experiment showed",
            "experimental results indicate",
            "researchers discovered",
            "study demonstrates",
            "model predicts",
            "quantum mechanics explains",
            "physics theory suggests"
        ],

        # =========================
        # MEDICAL / HEALTH CLAIMS
        # =========================
        "MEDICAL": [
            "causes cancer",
            "leads to autism",
            "associated with disease",
            "medical study found",
            "clinical trial showed",
            "vaccines cause",
            "no causal link was found",
            "health risks include",
            "doctors warn that"
        ],

        # =========================
        # SOCIAL / MISINFORMATION
        # =========================
        "SOCIAL": [
            "some blogs claim",
            "online forums suggest",
            "people believe that",
            "rumors say",
            "conspiracy theories claim",
            "misinformation spread online",
            "according to social media",
            "unverified sources claim"
        ]
    }

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model)

        # Precompute embeddings
        self.intent_embeddings = {
            intent: self.encoder.encode(
                examples, convert_to_tensor=True
            )
            for intent, examples in self.INTENTS.items()
        }

    def classify(self, claim):
        claim_emb = self.encoder.encode(claim, convert_to_tensor=True)

        best_intent = "UNKNOWN"
        best_score = 0.0

        for intent, emb in self.intent_embeddings.items():
            score = util.cos_sim(claim_emb, emb).max().item()
            if score > best_score:
                best_score = score
                best_intent = intent

        # Confidence gate (important)
        return best_intent if best_score >= 0.35 else "UNKNOWN"
