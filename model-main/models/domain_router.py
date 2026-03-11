class DomainRouter:
    def route(self, claim_text, entity_info):
        """
        Decide which domains/sources to use
        """
        text = claim_text.lower()

        # person / biography
        if entity_info["type"] == "person":
            return ["biography"]

        # medical
        if any(k in text for k in ["disease", "ulcer", "vaccine", "virus", "cancer"]):
            return ["medical"]

        # science / research
        if any(k in text for k in ["quantum", "physics", "theory", "algorithm"]):
            return ["science"]

        # fallback
        return ["general"]
