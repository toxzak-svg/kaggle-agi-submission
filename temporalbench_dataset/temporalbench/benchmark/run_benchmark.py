"""
TemporalBench — Run benchmark on an LLM
Local execution with OpenAI-compatible client.

Usage:
    python run_benchmark.py                    # v1, seed=0, local model
    python run_benchmark.py --version v3      # specific version
    python run_benchmark.py --all             # all versions × seeds
    python run_benchmark.py --model gpt-4     # specify model
"""

import argparse
import json
import time
from datetime import datetime

# Default LLM client (swap for kaggle_benchmarks in that environment)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from benchmark.benchmark import TemporalBench


class LLMClient:
    """Wraps OpenAI-compatible API for local/Kaggle use."""

    def __init__(self, model: str = "gpt-4", base_url: str = None, api_key: str = "ollama"):
        self.model = model
        if HAS_OPENAI:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        else:
            self.client = None

    def prompt(self, text: str) -> str:
        if self.client is None:
            # Dummy for testing without OpenAI SDK
            return "dummy_response"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            temperature=0.0,
            max_tokens=256,
        )
        return response.choices[0].message.content


def run_local(llm, args):
    """Run the benchmark locally."""
    bench = TemporalBench(versions=[args.version], seeds=[args.seed])

    print(f"\n{'='*60}")
    print(f" TemporalBench  |  version={args.version}  seed={args.seed}")
    print(f"{'='*60}\n")

    start = time.time()
    result = bench.run(llm, version=args.version, seed=args.seed)
    elapsed = time.time() - start

    task_results = result.get("task_results", {})

    print(f"Results (took {elapsed:.1f}s):\n")
    for task, res in task_results.items():
        if "error" in res:
            print(f"  {task:30s} ERROR: {res['error']}")
        else:
            acc = res.get("accuracy", res.get("staleness_error_rate", "N/A"))
            n = res.get("total", "?")
            print(f"  {task:30s}  {acc}  (n={n})")

    print(f"\n  TRS (composite): {result.get('TRS', 0.0):.4f}")
    print(f"\n  Full result saved to benchmark_results_{args.version}_s{args.seed}.json")
    return result


def run_all_versions(llm, args):
    """Run all version × seed combinations."""
    bench = TemporalBench()
    all_results = bench.run_all(llm)

    print(f"\n{'='*60}")
    print(" TemporalBench — Full Multi-Seed Run")
    print(f"{'='*60}\n")

    print(bench.summary_table(all_results))

    # Save
    out = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "results": {
            f"{v}_{s}": {
                "version": v,
                "seed": s,
                "TRS": r.get("TRS", 0.0),
                "task_results": r.get("task_results", {}),
            }
            for (v, s), r in all_results.items()
        },
    }
    fname = "benchmark_results_all.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {fname}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run TemporalBench on any OpenAI-compatible LLM")
    parser.add_argument("--version", default="v1", help="Benchmark version (v1, v2, v3, v4)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0, 1, 2)")
    parser.add_argument("--all", action="store_true", help="Run all version × seed combinations")
    parser.add_argument("--model", default="gpt-4", help="Model name")
    parser.add_argument("--base-url", default=None, help="API base URL (e.g. http://localhost:11434 for Ollama)")
    parser.add_argument("--api-key", default="ollama", help="API key")
    parser.add_argument("--output", default=None, help="Output JSON file")

    args = parser.parse_args()

    llm = LLMClient(model=args.model, base_url=args.base_url, api_key=args.api_key)

    if args.all:
        results = run_all_versions(llm, args)
    else:
        result = run_local(llm, args)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()