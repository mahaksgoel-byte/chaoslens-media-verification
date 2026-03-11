class Highlighter:
    def highlight(self, original_sentence, verdict):
        return {
            "text": original_sentence,
            "label": verdict
        }
