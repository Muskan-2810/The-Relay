
import csv
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def read_csv(filename):
    path = RESULTS / filename
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def percentile(values, p):
    values = sorted(float(x) for x in values if x not in ("", None))

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return values[lower] + fraction * (
        values[upper] - values[lower]
    )


def show_latency(label, rows):
    values = [
        float(r["total_ms"])
        for r in rows
        if r.get("total_ms")
    ]

    if not values:
        return

    print(f"\n{label}")
    print("Requests:", len(values))
    print("p50:", round(percentile(values, 50), 2), "ms")
    print("p95:", round(percentile(values, 95), 2), "ms")
    print("p99:", round(percentile(values, 99), 2), "ms")
    print("Min:", round(min(values), 2), "ms")
    print("Max:", round(max(values), 2), "ms")


def show_metrics(rows, fields):
    for field in fields:
        raw_values = [
            r[field] for r in rows
            if r.get(field, "") != ""
        ]

        if not raw_values:
            continue

        if raw_values[0].strip().lower() in ("true", "false"):
            correct = sum(
                value.strip().lower() == "true"
                for value in raw_values
            )

            percentage = correct / len(raw_values) * 100

            print(
                field,
                ":", correct, "/", len(raw_values),
                "=", round(percentage, 2), "%"
            )

        else:
            values = [float(value) for value in raw_values]

            print(
                field,
                "average:",
                round(statistics.mean(values), 4)
            )


def main():
    print("=" * 65)
    print("FINAL EXPERIMENT METRICS AUDIT")
    print("=" * 65)

    # 1. REAL GROQ RESULTS
    real = read_csv("real_results.csv")

    print("\n1. REAL GROQ API")
    print("Total requests:", len(real))
    print("Statuses:", dict(Counter(r["status"] for r in real)))

    successful = [
        r for r in real
        if r["status"] == "ok"
    ]

    print(
        "Success rate:",
        round(len(successful) / len(real) * 100, 2),
        "%"
    )

    show_latency("All real requests", real)
    show_latency("Successful real requests", successful)

    token_rows = [
        r for r in successful
        if r["tokens_in"] and r["tokens_out"]
    ]

    if token_rows:
        token_totals = [
            int(r["tokens_in"]) + int(r["tokens_out"])
            for r in token_rows
        ]

        print(
            "Average observed input + output tokens:",
            round(statistics.mean(token_totals), 2)
        )

    # 2. STAND-IN RESULTS
    standin = read_csv("standin_results.csv")
    grouped = defaultdict(list)

    for row in standin:
        grouped[row["scenario"]].append(row)

    print("\n\n2. STAND-IN SCENARIOS")

    for scenario, rows in grouped.items():
        show_latency(scenario, rows)

        print(
            "Statuses:",
            dict(Counter(r["status"] for r in rows))
        )

    # 3. RETRY RESULTS
    retries = read_csv("retry_results.csv")

    print("\n\n3. RETRY EXPERIMENT")

    for row in retries:
        print(
            row["scenario"],
            "| status:", row["status"],
            "| latency:", row["total_ms"],
            "| retries:", row["retries"]
        )

    # 4. RETRIEVAL SETTINGS
    retrieval = read_csv("retrieval_comparison.csv")
    settings = defaultdict(list)

    for row in retrieval:
        settings[row["setting"]].append(row)

    print("\n\n4. RETRIEVAL SETTINGS")

    retrieval_fields = (
        "top1_source_match",
        "top5_source_match",
        "concept_coverage",
        "latency_ms",
        "approx_retrieved_tokens"
    )

    for setting, rows in settings.items():
        print("\nSetting:", setting)
        print("Questions:", len(rows))

        show_metrics(rows, retrieval_fields)

    # 5. QUERY EXPANSION
    expansion = read_csv("query_expansion_comparison.csv")
    query_groups = defaultdict(list)

    for row in expansion:
        query_groups[row["query_type"]].append(row)

    print("\n\n5. QUERY EXPANSION")

    expansion_fields = (
        "top1_source_match",
        "top5_source_match",
        "concept_coverage",
        "latency_ms",
        "approx_retrieved_tokens"
    )

    for query_type, rows in query_groups.items():
        print("\nQuery type:", query_type)
        print("Questions:", len(rows))

        show_metrics(rows, expansion_fields)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("Original CSV files were not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
