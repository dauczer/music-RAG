"""Small, source-based retrieval evaluation for the French rap corpus.

Usage:
    python -m evals.run
    python -m evals.run --top-k 5 --output evals/runs/latest.json

The runner evaluates exact song metadata, not substrings found inside retrieved
documents. It requires HF_TOKEN because queries use the production embedding API.
"""

import argparse
import json
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from rag.vectorstore import list_artist_summaries, retrieve_chunks

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"


def _load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).casefold()


def _first_relevant_rank(retrieved_titles: list[str], expected_titles: list[str]) -> int | None:
    expected = {_normalize(title) for title in expected_titles}
    for rank, title in enumerate(retrieved_titles, 1):
        if _normalize(title) in expected:
            return rank
    return None


def run_eval(top_k: int) -> dict:
    questions = _load_questions()
    checks: list[dict] = []
    skipped = 0

    for case in questions:
        expected_sources = case.get("expected_sources", {})
        if not expected_sources:
            skipped += 1
            continue

        for artist, expected_titles in expected_sources.items():
            try:
                chunks = retrieve_chunks(artist, case["question"], n_results=top_k)
                retrieved_titles = [chunk.title for chunk in chunks]
                distances = [chunk.distance for chunk in chunks]
                error = None
            except (RuntimeError, ValueError) as exc:
                retrieved_titles = []
                distances = []
                error = str(exc)

            rank = _first_relevant_rank(retrieved_titles, expected_titles)
            checks.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "artist": artist,
                    "hit": rank is not None,
                    "rank": rank,
                    "reciprocal_rank": 1 / rank if rank else 0,
                    "expected_titles": expected_titles,
                    "retrieved": [
                        {"title": title, "distance": distance}
                        for title, distance in zip(retrieved_titles, distances, strict=True)
                    ],
                    "error": error,
                }
            )

    hits = sum(check["hit"] for check in checks)
    total = len(checks)
    mean_reciprocal_rank = sum(check["reciprocal_rank"] for check in checks) / total if total else 0

    category_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})
    for check in checks:
        bucket = category_totals[check["category"]]
        bucket["total"] += 1
        bucket["hits"] += int(check["hit"])

    summaries = list_artist_summaries()
    return {
        "run": {
            "created_at": datetime.now(UTC).isoformat(),
            "embedding_model": "intfloat/multilingual-e5-small",
            "top_k": top_k,
            "indexed_artists": len(summaries),
            "indexed_songs": sum(artist.song_count for artist in summaries),
            "indexed_chunks": sum(artist.chunk_count for artist in summaries),
        },
        "metrics": {
            f"hit_at_{top_k}": hits / total if total else 0,
            f"hits_at_{top_k}": hits,
            "retrieval_checks": total,
            f"mrr_at_{top_k}": mean_reciprocal_rank,
            "non_retrieval_cases": skipped,
            "by_category": dict(category_totals),
        },
        "checks": checks,
    }


def _print_report(report: dict) -> None:
    top_k = report["run"]["top_k"]
    print(f"{'Case':<28} {'Artist':<15} {'Rank':<6} Result")
    print("-" * 68)
    for check in report["checks"]:
        rank = str(check["rank"]) if check["rank"] else "—"
        result = "HIT" if check["hit"] else "MISS"
        if check["error"]:
            result = f"ERROR: {check['error']}"
        print(f"{check['case_id']:<28} {check['artist']:<15} {rank:<6} {result}")

    metrics = report["metrics"]
    print("-" * 68)
    print(
        f"Hit@{top_k}: {metrics[f'hits_at_{top_k}']}/{metrics['retrieval_checks']} "
        f"= {metrics[f'hit_at_{top_k}']:.0%}"
    )
    print(f"MRR@{top_k}: {metrics[f'mrr_at_{top_k}']:.3f}")
    print(f"Non-retrieval cases reserved for answer review: {metrics['non_retrieval_cases']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    report = run_eval(args.top_k)
    _print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
