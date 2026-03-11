from sentence_transformers import SentenceTransformer, util


class EntityContext:
    """
    Maintains rolling entity memory and resolves pronouns
    and implicit references.
    """

    PRONOUNS = {"he", "she", "they", "them", "his", "her", "their", "it"}

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.last_entity = None
        self.encoder = SentenceTransformer(model_name)

    def update(self, entity):
        if entity and entity.get("label"):
            self.last_entity = entity

    def resolve_pronoun(self, entity, claim_text):
        # Explicit entity → trust it
        if entity and entity.get("label"):
            return entity

        if not self.last_entity:
            return None

        tokens = set(claim_text.lower().split())

        # Pronoun detected → reuse last entity
        if tokens & self.PRONOUNS:
            return self.last_entity

        # Semantic continuity fallback
        claim_emb = self.encoder.encode(claim_text, convert_to_tensor=True)
        entity_emb = self.encoder.encode(
            self.last_entity["label"], convert_to_tensor=True
        )

        sim = util.cos_sim(claim_emb, entity_emb).item()
        if sim > 0.35:
            return self.last_entity

        return None
