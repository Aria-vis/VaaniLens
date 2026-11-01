"""Global configuration parameters for VaaniLens."""

import os

# Ensure models run fully offline using cached weights
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CONF_THRESHOLD = 0.75
LIME_SAMPLES = 200
MAX_TOKENS = 256
RANDOM_STATE = 42


