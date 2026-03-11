from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class Verifier:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # label mapping for MNLI-style models
        self.id2label = self.model.config.id2label

    def verify(self, claim, evidence):
        # MNLI expects: premise (evidence), hypothesis (claim)
        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0]

        # build readable score dict
        scores = {}
        for i, p in enumerate(probs):
            label = self.id2label[i].lower()
            scores[label] = float(p)

        return scores
