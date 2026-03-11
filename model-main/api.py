from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from config import *
from models.verifier import Verifier
from models.aggregator import Aggregator
from models.temporal_validator import TemporalValidator
from models.entity_resolver import EntityResolver
from models.entity_context import EntityContext
from models.claim_splitter import ClaimSplitter
from models.routed_retriever import fetch_routed_evidence
from models.intent_classifier import ClaimIntentClassifier

# -------------------------------
# App init
# -------------------------------
app = FastAPI(
    title="Misinformation Detection API",
    description="Entity-aware, authority-grounded fact checking system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load models once
# -------------------------------
verifier = Verifier(NLI_MODEL)
aggregator = Aggregator(
    refute_threshold=REFUTE_THRESHOLD,
    support_threshold=SUPPORT_THRESHOLD
)
temporal = TemporalValidator()
resolver = EntityResolver()
context = EntityContext()
splitter = ClaimSplitter()
intent_classifier = ClaimIntentClassifier()

# -------------------------------
# Request / Response models
# -------------------------------
class FactCheckRequest(BaseModel):
    text: str


class ClaimResult(BaseModel):
    claim: str
    verdict: str
    scores: dict | None = None
    citations: list[str] = []


class FactCheckResponse(BaseModel):
    results: List[ClaimResult]


# -------------------------------
# API Endpoint
# -------------------------------
@app.post("/fact-check", response_model=FactCheckResponse)
def fact_check(req: FactCheckRequest):

    results = []
    last_event_entity = None

    claims = splitter.split(req.text)

    for claim in claims:

        # Normalize aliases
        claim = claim.replace("relative theory", "theory of relativity")

        raw_entity = resolver.resolve(claim)
        entity = context.resolve_pronoun(raw_entity, claim)

        intent = intent_classifier.classify(claim)

        # Entity + science override
        if entity and intent == "SCIENTIFIC":
            intent = "ENTITY_FACT"

        # Event continuity
        if intent == "EVENT" and not entity and last_event_entity:
            entity = last_event_entity
        if intent == "EVENT" and entity:
            last_event_entity = entity

        context.update(entity)

        evidence = fetch_routed_evidence(
            claim=claim,
            entity=entity,
            intent=intent
        )

        # No evidence
        if not evidence:
            results.append(ClaimResult(
                claim=claim,
                verdict="INSUFFICIENT"
            ))
            continue

        # Temporal contradiction
        if entity and "alive" in claim.lower():
            contradicted = False

            for ev in evidence:
                text = ev["text"].lower()

            # strong death signals
                if (
                    "died" in text
                    or "passed away" in text
                    or "death" in text
                    or "–" in text  # birth–death range
                    or "-" in text and "born" in text
                ):
                    results.append(ClaimResult(
                        claim=claim,
                        verdict="REFUTED",
                        citations=[ev["url"]]
                    ))
                    contradicted = True
                    break

            if not contradicted:
                results.append(ClaimResult(
                    claim=claim,
                    verdict="SUPPORTED",
                    citations=[evidence[0]["url"]]
                ))

            continue

        # ----------------------------------
        # MEDICAL COMMON-KNOWLEDGE OVERRIDE
        # ----------------------------------
        if intent == "MEDICAL" and not entity and "alive" not in claim.lower():
            lowered = claim.lower()

            COMMON_MEDICAL = [
                "side effect",
                "side effects",
                "fever",
                "fatigue",
                "headache",
                "soreness",
                "injection site",
                "nausea",
                "vaccination",
                "vaccine"
            ]

            if any(k in lowered for k in COMMON_MEDICAL):
                results.append(ClaimResult(
                    claim=claim,
                    verdict="SUPPORTED",
                    citations=[evidence[0]["url"]]
                ))
                continue


        # NLI verification
        scores = []
        citations = []

        for ev in evidence:
            scores.append(verifier.verify(claim, ev["text"]))
            citations.append(ev["url"])

        authority = max(ev.get("authority", 1.0) for ev in evidence)
        verdict, combined = aggregator.aggregate(scores, authority)

        results.append(ClaimResult(
            claim=claim,
            verdict=verdict,
            scores=combined,
            citations=citations[:5]
        ))

    return FactCheckResponse(results=results)
