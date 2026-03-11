class Aggregator:
    """
    Aggregates NLI scores across evidence and decides final verdict.
    Supports authority-aware entailment boosting.
    """

    def __init__(
        self,
        support_threshold=0.7,
        refute_threshold=0.7,
        authority_boost=0.05
    ):
        self.support_threshold = support_threshold
        self.refute_threshold = refute_threshold
        self.authority_boost = authority_boost

    def aggregate(self, scores_list, source_authority=1.0, intent=None):
        if not scores_list:
            return "INSUFFICIENT", None

        avg = {
            "contradiction": 0.0,
            "neutral": 0.0,
            "entailment": 0.0
        }

        for scores in scores_list:
            for k in avg:
                avg[k] += scores.get(k, 0.0)

        for k in avg:
            avg[k] /= len(scores_list)

        # Authority boost
        if source_authority >= 0.85:
            avg["entailment"] = min(avg["entailment"] + self.authority_boost, 1.0)

        # Strong refutation
        if avg["contradiction"] >= self.refute_threshold:
            return "REFUTED", avg

        # Strong support
        if avg["entailment"] >= self.support_threshold:
            return "SUPPORTED", avg

        # 🟡 PARTIAL SUPPORT (NEW)
        if intent in {"MEDICAL", "SCIENTIFIC", "TECH"}:
            if avg["entailment"] >= 0.45 and avg["contradiction"] < 0.3:
                return "PARTIALLY_SUPPORTED", avg

        return "INSUFFICIENT", avg
