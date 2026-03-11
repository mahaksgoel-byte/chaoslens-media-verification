import sys

from config import *
from models.verifier import Verifier
from models.aggregator import Aggregator
from models.temporal_validator import TemporalValidator
from models.entity_resolver import EntityResolver
from models.entity_context import EntityContext
from models.claim_splitter import ClaimSplitter
from models.routed_retriever import fetch_routed_evidence
from models.intent_classifier import ClaimIntentClassifier


def main():
    verifier = Verifier(NLI_MODEL)
    aggregator = Aggregator(
        refute_threshold=REFUTE_THRESHOLD,
        support_threshold=SUPPORT_THRESHOLD
    )
    temporal = TemporalValidator()

    splitter = ClaimSplitter()
    resolver = EntityResolver()
    context = EntityContext()
    intent_classifier = ClaimIntentClassifier()

    last_event_entity = None

    print("\nEnter text to fact-check:\n")
    print("(Press Ctrl+D on Mac/Linux or Ctrl+Z then Enter on Windows)")
    text = sys.stdin.read()

    print("\n--- ANALYSIS START ---\n")

    claims = splitter.split(text)

    for idx, claim in enumerate(claims, 1):

        # -------------------------------
        # Alias normalization
        # -------------------------------
        ALIASES = {
            "relative theory": "theory of relativity"
        }
        for k, v in ALIASES.items():
            claim = claim.replace(k, v)

        print(f"\nClaim {idx}: {claim}")

        raw_entity = resolver.resolve(claim)
        entity = context.resolve_pronoun(raw_entity, claim)

        intent = intent_classifier.classify(claim)

        # -------------------------------
        # ENTITY + SCIENCE OVERRIDE
        # -------------------------------
        if entity and intent == "SCIENTIFIC":
            intent = "ENTITY_FACT"

        # -------------------------------
        # EVENT CONTINUITY
        # -------------------------------
        if intent == "EVENT" and not entity and last_event_entity:
            entity = last_event_entity

        if intent == "EVENT" and entity:
            last_event_entity = entity

        context.update(entity)

        print(f"[DEBUG] Entity: {entity}")
        print(f"[DEBUG] Intent: {intent}")

        evidence = fetch_routed_evidence(
            claim=claim,
            entity=entity,
            intent=intent
        )

        if not evidence:
            print("Verdict: INSUFFICIENT")
            print("Reason: No authoritative evidence found.")
            continue

        # ===============================
        # TEMPORAL CONTRADICTION CHECK
        # ===============================
        if entity and "alive" in claim.lower():
            contradicted = False
            for ev in evidence:
                if "died" in ev["text"].lower():
                    print("Verdict: REFUTED")
                    print("Reason: Authoritative source confirms death.")
                    print("Citations:")
                    print(f"- [{ev['source']}] {ev['title']}")
                    print(f"  {ev['url']}")
                    contradicted = True
                    break

            if contradicted:
                continue

        # ===============================
        # NLI VERIFICATION
        # ===============================
        scores = []
        for ev in evidence:
            s = verifier.verify(claim, ev["text"])
            scores.append(s)

        authority = max(ev.get("authority", 1.0) for ev in evidence)
        verdict, combined = aggregator.aggregate(scores, authority, intent=intent)

        print(f"Verdict: {verdict}")
        print(f"Scores: {combined}")

        print("Citations:")
        for ev in evidence[:5]:
            print(f"- [{ev['source']}] {ev['title']} ({ev.get('year')})")
            print(f"  {ev['url']}")

    print("\n--- ANALYSIS END ---\n")


if __name__ == "__main__":
    main()
