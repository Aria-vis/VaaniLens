"""Global configuration parameters for VaaniLens."""

import os

if os.environ.get("VAANI_OFFLINE") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CONF_THRESHOLD = 0.75

LIME_SAMPLES = 200

MAX_TOKENS = 256

RANDOM_STATE = 42


