#!/usr/bin/env python3
"""Compare lexical similarity and diversity of two prompt-pair JSONL files.

The core audit uses only the Python standard library, so it requires no model
download.  When NumPy is available it also reports the Vendi diversity score.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
import zlib
from pathlib import Path

try:
    import numpy as np
except ImportError:  # The remaining audit is intentionally dependency-free.
    np = None


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            q = row.get("original_query", row.get("q"))
            p = row.get("privileged_prompt", row.get("p"))
            sample_id = row.get("source_sample_id", row.get("sample_id"))
            if not all(isinstance(value, str) and value for value in (q, p, sample_id)):
                raise ValueError(f"{path}:{line_number}: missing q, p, or sample ID")
            rows.append({"id": sample_id, "q": q, "p": p})
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
    }


def features(text: str, analyzer: str) -> collections.Counter:
    if analyzer == "word12":
        toks = tokens(text)
        return collections.Counter(ngrams(toks, 1) + ngrams(toks, 2))
    if analyzer == "char35":
        chars = " ".join(text.lower().split())
        return collections.Counter(
            chars[index : index + n]
            for n in range(3, 6)
            for index in range(len(chars) - n + 1)
        )
    raise ValueError(analyzer)


def tfidf_vectors(texts: list[str], analyzer: str) -> list[dict]:
    counts = [features(text, analyzer) for text in texts]
    document_frequency = collections.Counter()
    for counter in counts:
        document_frequency.update(counter.keys())
    size = len(texts)
    result = []
    for counter in counts:
        vector = {
            term: count * (math.log((1 + size) / (1 + document_frequency[term])) + 1)
            for term, count in counter.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        result.append({term: value / norm for term, value in vector.items()})
    return result


def cosine(left: dict, right: dict) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def pairwise(vectors: list[dict]) -> list[float]:
    return [
        cosine(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]


def nearest_neighbor(vectors: list[dict]) -> list[float]:
    return [
        max(cosine(vector, other) for j, other in enumerate(vectors) if i != j)
        for i, vector in enumerate(vectors)
    ]


def cross_nearest(left: list[dict], right: list[dict]) -> list[float]:
    return [max(cosine(vector, other) for other in right) for vector in left]


def vendi_score(vectors: list[dict]) -> dict[str, float] | None:
    if np is None:
        return None
    size = len(vectors)
    gram = np.eye(size, dtype=np.float64)
    for i in range(size):
        for j in range(i + 1, size):
            gram[i, j] = gram[j, i] = cosine(vectors[i], vectors[j])
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0)
    probabilities = eigenvalues / eigenvalues.sum()
    probabilities = probabilities[probabilities > 1e-15]
    score = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {"score": score, "normalized_by_n": score / size}


def corpus_distinct(texts: list[str], n: int) -> float:
    all_ngrams = [item for text in texts for item in ngrams(tokens(text), n)]
    return len(set(all_ngrams)) / len(all_ngrams)


def mean_document_distinct(texts: list[str], n: int) -> float:
    ratios = []
    for text in texts:
        items = ngrams(tokens(text), n)
        ratios.append(len(set(items)) / len(items))
    return statistics.fmean(ratios)


def compression_ratio(texts: list[str]) -> float:
    raw = "\n".join(texts).encode()
    return len(zlib.compress(raw, level=9)) / len(raw)


def jaccard(left: str, right: str) -> float:
    left_set, right_set = set(tokens(left)), set(tokens(right))
    return len(left_set & right_set) / len(left_set | right_set)


def js_divergence(left_texts: list[str], right_texts: list[str]) -> float:
    left = collections.Counter(token for text in left_texts for token in tokens(text))
    right = collections.Counter(token for text in right_texts for token in tokens(text))
    left_total, right_total = sum(left.values()), sum(right.values())
    result = 0.0
    for term in left.keys() | right.keys():
        p = left[term] / left_total
        q = right[term] / right_total
        midpoint = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def top_document_ngrams(texts: list[str], n: int, limit: int = 12) -> list[dict]:
    frequency = collections.Counter()
    for text in texts:
        frequency.update(set(ngrams(tokens(text), n)))
    return [
        {"ngram": " ".join(term), "documents": count, "share": count / len(texts)}
        for term, count in frequency.most_common(limit)
    ]


def category_stats(rows: list[dict], manifest: Path | None) -> dict | None:
    if manifest is None:
        return None
    categories = {}
    for line in manifest.open(encoding="utf-8"):
        row = json.loads(line)
        categories[row["source_sample_id"]] = row["source_metadata"]["tag"]
    counts = collections.Counter(categories[row["id"]] for row in rows)
    probabilities = [count / len(rows) for count in counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    normalized = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return {"counts": dict(sorted(counts.items())), "normalized_entropy": normalized}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    old_rows = load_rows(args.old)
    new_rows = load_rows(args.new)
    old_by_id = {row["id"]: row for row in old_rows}
    new_by_id = {row["id"]: row for row in new_rows}
    shared_ids = sorted(old_by_id.keys() & new_by_id.keys())
    all_p = [row["p"] for row in old_rows + new_rows]
    old_size = len(old_rows)

    result = {
        "inputs": {
            "old": {"path": str(args.old), "sha256": sha256(args.old), "rows": len(old_rows)},
            "new": {"path": str(args.new), "sha256": sha256(args.new), "rows": len(new_rows)},
            "shared_sample_ids": len(shared_ids),
        },
        "length_words": {
            "old": describe([len(tokens(row["p"])) for row in old_rows]),
            "new": describe([len(tokens(row["p"])) for row in new_rows]),
        },
        "exact_duplicates": {
            "old_duplicate_rows": len(old_rows) - len({row["p"].strip().lower() for row in old_rows}),
            "new_duplicate_rows": len(new_rows) - len({row["p"].strip().lower() for row in new_rows}),
            "cross_exact_prompt_matches": len(
                {row["p"].strip().lower() for row in old_rows}
                & {row["p"].strip().lower() for row in new_rows}
            ),
        },
        "lexical_distribution": {
            "unigram_js_divergence_bits_0_identical_1_disjoint": js_divergence(
                [row["p"] for row in old_rows], [row["p"] for row in new_rows]
            ),
            "old_top_4grams": top_document_ngrams([row["p"] for row in old_rows], 4),
            "new_top_4grams": top_document_ngrams([row["p"] for row in new_rows], 4),
        },
        "diversity": {},
        "matched_same_q": {},
        "category_coverage": {
            "old": category_stats(old_rows, args.manifest),
            "new": category_stats(new_rows, args.manifest),
        },
    }

    for label, rows in (("old", old_rows), ("new", new_rows)):
        texts = [row["p"] for row in rows]
        result["diversity"][label] = {
            "corpus_distinct_1": corpus_distinct(texts, 1),
            "corpus_distinct_2": corpus_distinct(texts, 2),
            "mean_document_distinct_1": mean_document_distinct(texts, 1),
            "mean_document_distinct_2": mean_document_distinct(texts, 2),
            "zlib_ratio_lower_more_repetitive": compression_ratio(texts),
        }

    joint_word_new_vectors = None
    for analyzer in ("word12", "char35"):
        vectors = tfidf_vectors(all_p, analyzer)
        old_vectors, new_vectors = vectors[:old_size], vectors[old_size:]
        if analyzer == "word12":
            joint_word_new_vectors = new_vectors
        result["diversity"]["old"][f"{analyzer}_within_pair_cosine"] = describe(pairwise(old_vectors))
        result["diversity"]["new"][f"{analyzer}_within_pair_cosine"] = describe(pairwise(new_vectors))
        result["diversity"]["old"][f"{analyzer}_nearest_neighbor_cosine"] = describe(
            nearest_neighbor(old_vectors)
        )
        result["diversity"]["new"][f"{analyzer}_nearest_neighbor_cosine"] = describe(
            nearest_neighbor(new_vectors)
        )
        result["diversity"]["old"][f"{analyzer}_vendi"] = vendi_score(old_vectors)
        result["diversity"]["new"][f"{analyzer}_vendi"] = vendi_score(new_vectors)
        result.setdefault("cross_corpus", {})[f"new_to_old_{analyzer}_nearest_cosine"] = describe(
            cross_nearest(new_vectors, old_vectors)
        )

        paired_old = [old_vectors[list(old_by_id).index(sample_id)] for sample_id in shared_ids]
        paired_new = [new_vectors[list(new_by_id).index(sample_id)] for sample_id in shared_ids]
        result["matched_same_q"][f"old_vs_new_p_{analyzer}_cosine"] = describe(
            [cosine(left, right) for left, right in zip(paired_old, paired_new)]
        )
        result["matched_same_q"][f"old_{analyzer}_within_pair_cosine"] = describe(pairwise(paired_old))
        result["matched_same_q"][f"new_{analyzer}_within_pair_cosine"] = describe(pairwise(paired_new))
        result["matched_same_q"][f"old_{analyzer}_nearest_neighbor_cosine"] = describe(
            nearest_neighbor(paired_old)
        )
        result["matched_same_q"][f"new_{analyzer}_nearest_neighbor_cosine"] = describe(
            nearest_neighbor(paired_new)
        )
        result["matched_same_q"][f"old_{analyzer}_vendi"] = vendi_score(paired_old)
        result["matched_same_q"][f"new_{analyzer}_vendi"] = vendi_score(paired_new)

    paired_old_rows = [old_by_id[sample_id] for sample_id in shared_ids]
    paired_new_rows = [new_by_id[sample_id] for sample_id in shared_ids]
    result["matched_same_q"]["old_vs_new_p_token_jaccard"] = describe(
        [jaccard(left["p"], right["p"]) for left, right in zip(paired_old_rows, paired_new_rows)]
    )

    qp_texts = []
    qp_pairs = []
    for label, rows in (("old", old_rows), ("new", new_rows)):
        for row in rows:
            q_index = len(qp_texts)
            qp_texts.extend((row["q"], row["p"]))
            qp_pairs.append((label, q_index, q_index + 1))
    qp_vectors = tfidf_vectors(qp_texts, "word12")
    for label in ("old", "new"):
        values = [cosine(qp_vectors[q], qp_vectors[p]) for item_label, q, p in qp_pairs if item_label == label]
        result["diversity"][label]["q_to_p_word12_cosine"] = describe(values)

    rng = random.Random(42)
    bootstrap = {"new_equal_n": old_size, "replicates": args.bootstrap}
    samples = collections.defaultdict(list)
    assert joint_word_new_vectors is not None
    for _ in range(args.bootstrap):
        indices = rng.sample(range(len(new_rows)), old_size)
        texts = [new_rows[index]["p"] for index in indices]
        vectors = [joint_word_new_vectors[index] for index in indices]
        samples["corpus_distinct_1"].append(corpus_distinct(texts, 1))
        samples["corpus_distinct_2"].append(corpus_distinct(texts, 2))
        samples["nearest_neighbor_cosine"].append(statistics.fmean(nearest_neighbor(vectors)))
        samples["zlib_ratio"].append(compression_ratio(texts))
    for key, values in samples.items():
        bootstrap[key] = {
            "mean": statistics.fmean(values),
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
        }
    result["diversity"]["new_equal_size_bootstrap"] = bootstrap

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
