# baseline/run.py
# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point for the baseline evaluation.
# Called by inference.py — can also be run directly for testing.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()
print(f"DEBUG TOKEN: '{os.environ.get('HF_TOKEN', 'NOT SET')[:10]}...'")

# If HF_TOKEN not set in .env, fall back to the HF CLI cache file
if not os.environ.get("HF_TOKEN"):
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            os.environ["HF_TOKEN"] = f.read().strip()
        print(f"✓ Loaded HF_TOKEN from cache: {os.environ['HF_TOKEN'][:8]}...")

from baseline.evaluator import run_evaluation

def main():
    base_url = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
    scores   = run_evaluation(base_url=base_url, verbose=True)
    return scores


if __name__ == "__main__":
    main()