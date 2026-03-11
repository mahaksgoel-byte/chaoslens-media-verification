# config.py

# -------- MODELS --------
CLAIM_MODEL = "t5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "facebook/bart-large-mnli"

# -------- RETRIEVAL --------
TOP_K_EVIDENCE = 5

# -------- VERDICT THRESHOLDS --------
# Tuned for BART-large-MNLI + real-world evidence
SUPPORT_THRESHOLD = 0.7
REFUTE_THRESHOLD = 0.7

# -------- AUTHORITY BOOST --------
# Applied when evidence comes from high-authority sources
AUTHORITY_ENTAILMENT_BOOST = 0.05
