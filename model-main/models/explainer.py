class Explainer:
    def explain(self, claim, verdict, evidence):
        return {
            "claim": claim,
            "verdict": verdict,
            "reason": f"Based on retrieved evidence, this claim is {verdict.lower()}",
            "citations": evidence
        }
