import os


NEO4J_URI = "bolt://xxx"
NEO4J_USER = "xxx"
NEO4J_PASSWORD = "xxx"

MONGO_URI = "mongodb://xxx"
MONGO_DB_NAME = "SLCdb"
MONGO_COLLECTION_NAME = "papers"

# Sentence-BERT model name and output dimension
# Use all-MiniLM-L6-v2 (384 dim) for faster inference; can switch to specter2_base (768 dim) later if needed
# SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_MODEL_NAME = "all-mpnet-base-v2"
SBERT_DIM = 768


# Optional configurable schema details (match actual graph node labels)
LABEL_SLC = "SLCGene"
LABEL_PATHWAY = "Pathway"
LABEL_DISEASE = "Disease"
LABEL_RELA_EVENT = "RelaEvent"  # NEW: Event node for N-ary relations
REL_DOI_FIELD = "paper_dois"
REL_BELONGS_TO = "belongs_to"

# Node name field in Neo4j
NODE_NAME_FIELD = "name"

# MongoDB collections for evidence
MONGO_COLLECTION_PAPERS = "papers"
MONGO_COLLECTION_PW_DZ = "pathway_disease"
MONGO_COLLECTION_SLC_PW = "slc_pathway"

# Field names inside pair-evidence collections
MONGO_FIELD_SLC = "slc"
MONGO_FIELD_PATHWAY = "pathway"
MONGO_FIELD_DISEASE = "disease"

# Neighbor aggregation for SLC node
NEIGHBOR_AGG_METHOD = "mean"  # "mean", "sum", or "max"


# LLM summarization settings
SUMMARY_OUTPUT_DIR = os.environ.get("SUMMARY_OUTPUT_DIR", "data/precomputed_summaries")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "claude-sonnet-4-5-20250929")
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", 512))

