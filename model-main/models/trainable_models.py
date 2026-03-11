#!/usr/bin/env python3
"""
Enhanced model classes that support loading trained models
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
import os

class TrainableVerifier:
    """Enhanced Verifier that can load fine-tuned models"""
    
    def __init__(self, model_name=None, use_trained=True):
        # Use trained model if available
        if use_trained and os.path.exists("/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_nli"):
            model_path = "/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_nli"
            print(f"Loading fine-tuned NLI model from {model_path}")
        else:
            model_path = model_name or "facebook/bart-large-mnli"
            print(f"Using base NLI model: {model_path}")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def verify(self, claim, evidence):
        inputs = self.tokenizer(
            evidence, claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0]

        scores = {}
        for i, p in enumerate(probs):
            label = self.id2label[i].lower()
            scores[label] = float(p)

        return scores

class TrainableIntentClassifier:
    """Enhanced Intent Classifier that can load trained models"""
    
    def __init__(self, model=None, use_trained=True):
        # Use trained model if available
        if use_trained and os.path.exists("/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_intent"):
            model_path = "/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_intent"
            print(f"Loading fine-tuned intent model from {model_path}")
            self.encoder = SentenceTransformer(model_path)
        else:
            model_path = model or "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Using base intent model: {model_path}")
            self.encoder = SentenceTransformer(model_path)

        # Intent definitions (same as original)
        self.INTENTS = {
            "ENTITY_FACT": [
                "was born in", "was a famous scientist", "is a singer",
                "won the Nobel Prize", "died in", "is still alive"
            ],
            "EVENT": [
                "was shut down", "incident occurred", "was destroyed",
                "officials confirmed", "accident happened"
            ],
            "SCIENTIFIC": [
                "developed a theory", "experiment showed", "researchers discovered",
                "study demonstrates", "physics theory suggests"
            ],
            "MEDICAL": [
                "causes cancer", "medical study found", "vaccines cause",
                "health risks include", "doctors warn that"
            ],
            "SOCIAL": [
                "some blogs claim", "online forums suggest", "people believe that",
                "rumors say", "misinformation spread online"
            ]
        }

        # Precompute embeddings
        self.intent_embeddings = {
            intent: self.encoder.encode(examples, convert_to_tensor=True)
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

        return best_intent if best_score >= 0.35 else "UNKNOWN"

class TrainableClaimSplitter:
    """Enhanced Claim Splitter that can load trained T5 models"""
    
    def __init__(self, model_name=None, use_trained=True):
        # Use trained model if available
        if use_trained and os.path.exists("/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_splitter"):
            model_path = "/Users/mehakgoel/Desktop/Bluebit/model-main/models/trained_splitter"
            print(f"Loading fine-tuned claim splitter from {model_path}")
        else:
            model_path = model_name or "t5-small"
            print(f"Using base claim splitter: {model_path}")
            
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model.eval()

    def split(self, text):
        # Use T5 to split text into claims
        input_text = f"split claims: {text}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode and split claims
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        claims = [claim.strip() for claim in result.split("[CLAIM_SEP]") if claim.strip()]
        
        return claims if claims else [text]  # Fallback to original text
