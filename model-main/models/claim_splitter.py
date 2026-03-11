import re

class ClaimSplitter:
    """
    Robust claim splitter:
    - Handles paragraphs
    - Preserves sentence boundaries
    - Does NOT drop short but meaningful claims
    """

    def split(self, text):
        if not text or not text.strip():
            return []

        # Normalize whitespace
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)

        claims = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                claims.append(sent)

        return claims
