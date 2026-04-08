import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

# Fall back to the HuggingFace CLI token cache if HF_TOKEN isn't in .env
if not os.environ.get("HF_TOKEN"):
    cache_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            os.environ["HF_TOKEN"] = f.read().strip()

from baseline.evaluator import run_evaluation


def main():
    base_url = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
    return run_evaluation(base_url=base_url, verbose=True)


if __name__ == "__main__":
    main()
