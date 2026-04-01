# inference.py
# ─────────────────────────────────────────────────────────────────────────────
# Root-level entry point for hackathon evaluation.
# Judges run this file to verify reproducible scores across all three tasks.
#
# Required environment variables:
#   API_BASE_URL  — LLM API endpoint
#   MODEL_NAME    — Model identifier for inference
#   HF_TOKEN      — Hugging Face / API key
#   ENV_BASE_URL  — Running environment server URL (default: localhost:7860)
#
# Usage:
#   python inference.py
#
# Runtime must be under 20 minutes on 2vCPU / 8GB RAM.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline.run import main


if __name__ == "__main__":
    scores = main()

    # Machine-readable summary for auto-validator
    print("\nSCORES:")
    print(f"  easy:    {scores.get('easy',    0.0):.4f}")
    print(f"  medium:  {scores.get('medium',  0.0):.4f}")
    print(f"  hard:    {scores.get('hard',    0.0):.4f}")
    print(f"  average: {scores.get('average', 0.0):.4f}")

    # Warn and exit non-zero if evaluation failed entirely
    if scores.get("average", 0.0) == 0.0:
        print("\nWARNING: All scores are zero. Check:")
        print("  1. Is the environment server running?")
        print(f"     ENV_BASE_URL = {os.environ.get('ENV_BASE_URL', 'http://localhost:7860')}")
        print("  2. Are API credentials set?")
        print(f"     API_BASE_URL = {os.environ.get('API_BASE_URL', 'NOT SET')}")
        print(f"     MODEL_NAME   = {os.environ.get('MODEL_NAME',   'NOT SET')}")
        print(f"     HF_TOKEN     = {'SET' if os.environ.get('HF_TOKEN') else 'NOT SET'}")
        sys.exit(1)