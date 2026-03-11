from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class ClaimExtractor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def extract_claims(self, text):
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        claims = []

        for sent in sentences:
            prompt = f"Extract a factual claim: {sent}"
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4
                )

            claim = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

            claim = claim.replace("Extract a factual claim:", "").replace("extract a factual claim:", "").strip(" :")

            if len(claim.split()) < 3:
                claim = sent

            claims.append({
                "original": sent,
                "claim": claim
            })

        return claims
