from collections import defaultdict

class TemporalValidator:
    def __init__(self, year_gap=8):
        """
        year_gap: minimum gap between old and new research
        """
        self.year_gap = year_gap

    def split_by_time(self, evidence):
        """
        Split evidence into old and new buckets
        """
        years = [e["year"] for e in evidence if e.get("year")]
        if not years:
            return [], []

        max_year = max(years)
        cutoff = max_year - self.year_gap

        old = []
        new = []

        for e in evidence:
            y = e.get("year")
            if not y:
                continue
            if y <= cutoff:
                old.append(e)
            else:
                new.append(e)

        return old, new

    def detect_outdated(self, claim, old_scores, new_scores):
        """
        old_scores / new_scores: lists of NLI score dicts
        """

        if not old_scores or not new_scores:
            return False, None

        def avg(scores, key):
            return sum(s[key] for s in scores) / len(scores)

        old_support = avg(old_scores, "entailment")
        new_support = avg(new_scores, "entailment")

        old_contra = avg(old_scores, "contradiction")
        new_contra = avg(new_scores, "contradiction")

        # classic outdated pattern
        if old_support > 0.6 and new_contra > 0.6:
            return True, "Newer research contradicts older conclusions."

        return False, None
