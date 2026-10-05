#!/usr/bin/env python3
"""Score WISE images with the official binary judge protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.benchmarks.common import (ROOT, EVALUATOR_ASSETS, clean_env, collect_records, digest, evaluator_root,
                                          read_jsonl, run_logged, task_main, write_json,
                                          write_jsonl)

FILES = (
    "cultural_common_sense_verified.json",
    "spatio-temporal_reasoning_verified.json",
    "natural_science_verified.json",
)
WEIGHTS = {"CULTURE": .40, "TIME": .12, "SPACE": .12,
           "BIOLOGY": .12, "PHYSICS": .12, "CHEMISTRY": .12}


def category(pid):
    if type(pid) is not int or not 1 <= pid <= 1000:
        raise ValueError(f"Invalid WISE prompt ID: {pid}")
    return "CULTURE" if pid <= 400 else tuple(WEIGHTS)[1 + (pid - 401) // 120]


def load_official(upstream):
    upstream = upstream.resolve()
    records = []
    for filename in FILES:
        path = upstream / "data_verified" / filename
        for index, row in enumerate(json.loads(path.read_text())):
            for field in ("Prompt", "Explanation", "Category", "Subcategory"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Invalid {field}: {filename}:{index}")
            category(row["prompt_id"])
            records.append((row, path, index))
    ids = [row["prompt_id"] for row, _, _ in records]
    if len(ids) != 1000 or set(ids) != set(range(1, 1001)):
        raise ValueError("WISE must contain each official ID 1..1000 exactly once")
    return sorted(records, key=lambda item: item[0]["prompt_id"])


def summarize(rows, scores):
    expected = {r["source_metadata"]["prompt_id"] for r in rows}
    if len(expected) != len(rows) or not expected:
        raise ValueError("Manifest has empty or duplicate IDs")
    by_id = {}
    for score in scores:
        pid = score["prompt_id"]
        if pid in by_id or pid not in expected or score.get("score") not in (0, 1):
            raise ValueError("Duplicate, unexpected, or invalid WISE score")
        by_id[pid] = score["score"]
    complete = set(by_id) == expected
    categories = {}
    for name in WEIGHTS:
        ids = {pid for pid in expected if category(pid) == name}
        values = [by_id[pid] for pid in ids if pid in by_id]
        categories[name] = dict(expected=len(ids), scored=len(values),
                                mean=sum(values) / len(values) if values else None)
    weighted = None
    if complete and all(c["expected"] for c in categories.values()):
        weighted = sum(WEIGHTS[name] * categories[name]["mean"] for name in WEIGHTS)
    return dict(status="completed" if complete else "incomplete", expected=len(expected),
                scored=len(by_id), missing_ids=sorted(expected - set(by_id)),
                categories=categories, wiscore=weighted,
                scope="official-full" if expected == set(range(1, 1001)) else "internal-split",
                protocol="WISE_Verified")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--agent-run", type=Path)
    source.add_argument("--image-dir", type=Path, help="Official <prompt_id>.png filenames")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--upstream", type=Path, default=EVALUATOR_ASSETS / "wise")
    p.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen3.5-35B-A3B")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--preflight", action="store_true", help="Validate inputs without judge calls")
    p.add_argument("--fail-fast", action="store_true", help="Stop on exhausted official retries; never score errors as zero")
    a = p.parse_args()
    if a.workers < 1 or a.timeout < 1:
        raise ValueError("Workers and timeout must be positive")
    rows = read_jsonl(a.manifest)
    summarize(rows, [])
    official = {row["prompt_id"]: row for row, _, _ in load_official(a.upstream)}
    for row in rows:
        pid = row["source_metadata"]["prompt_id"]
        if row["dataset_id"] != "wise" or row["prompt"] != official[pid]["Prompt"]:
            raise ValueError("Manifest does not match WISE prompts")
    if a.agent_run:
        splits = {r["split"] for r in rows}
        if len(splits) != 1:
            raise ValueError("Agent evaluation requires one split")
        records, _ = collect_records(benchmark="wise", split=splits.pop(),
            manifest=a.manifest.resolve(), agent_run=a.agent_run.resolve(), expected_count=len(rows))
        images = {r["source"]["source_metadata"]["prompt_id"]: Path(r["selected_image_path"])
                  for r in records}
    else:
        images = {r["source_metadata"]["prompt_id"]:
                  a.image_dir.resolve() / f"{r['source_metadata']['prompt_id']}.png" for r in rows}
    for path in images.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty image: {path}")
    identity = dict(manifest_sha256=digest(a.manifest), model=a.model, api_base=a.api_base,
                    upstream_code_sha256=digest(a.upstream / "vllm_eval.py"),
                    reference_sha256={str(pid): __import__('hashlib').sha256(
                        json.dumps(official[pid], sort_keys=True).encode()).hexdigest() for pid in images},
                    image_sha256={str(pid): digest(path) for pid, path in images.items()})
    if a.preflight:
        print(json.dumps(dict(status="ready", count=len(rows), identity=identity)))
        return 0
    a.output_dir.mkdir(parents=True, exist_ok=True)
    seal = a.output_dir / "identity.json"
    if seal.exists() and json.loads(seal.read_text()) != identity:
        raise ValueError("Evaluation inputs/protocol changed; use a new output directory")
    if not seal.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Refusing to adopt unsealed evaluation outputs")
    write_json(seal, identity)
    score_path = a.output_dir / "scores.jsonl"
    scores = read_jsonl(score_path) if score_path.exists() else []
    summarize(rows, scores)
    done = {s["prompt_id"] for s in scores}
    spec = importlib.util.spec_from_file_location("wise_official", a.upstream / "vllm_eval.py")
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    cfg = dict(api_base=a.api_base.rstrip('/'), api_key=os.getenv("VLLM_API_KEY", ""),
               model=a.model, timeout=a.timeout)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(judge.evaluate_image, pid, official[pid], str(path), cfg)
                   for pid, path in images.items() if pid not in done]
        for future in as_completed(futures):
            result = future.result()
            if a.fail_fast and result.get("status") != "ok":
                for pending in futures:
                    pending.cancel()
                raise RuntimeError("WISE official judge retries exhausted; evaluation stopped")
            if result.get("status") == "ok":
                scores.append(result["score"])
                write_json(a.output_dir / "responses" / f"{result['score']['prompt_id']}.json", result["full"])
                write_jsonl(score_path, sorted(scores, key=lambda r: r["prompt_id"]))
            write_json(a.output_dir / "summary.json", summarize(rows, scores))
    summary = summarize(rows, scores)
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "completed" else 1


def score(task: dict, env: dict[str, str]) -> None:
    dest = Path(task["evaluation_dir"])
    scores = dest / "scores"
    judge = task["judge"]
    log = dest / "metadata" / "evaluator.log"
    run_logged([
        sys.executable, str(Path(__file__).resolve()),
        "--manifest", str(task["manifest"]),
        "--image-dir", str(dest / "images"),
        "--output-dir", str(scores),
        "--upstream", str(evaluator_root(task, "wise")),
        "--api-base", str(judge["api_base"]),
        "--model", str(judge["model"]),
        "--workers", str(judge.get("workers", 1)),
        "--fail-fast",
    ], cwd=ROOT, log=log, env=env)
    summary = json.loads((scores / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("scored") != task["expected_count"]:
        raise ValueError("WISE score coverage is incomplete")
    write_json(scores / "metrics.json", dict(summary, benchmark="wise",
                                             sample_count=summary["scored"]))


def prepare_vllm_env(env: dict[str, str], *, bin_dir: Path) -> dict[str, str]:
    result = dict(env)
    cuda = Path(result.get("CUDA_HOME") or "/usr/local/cuda")
    result["PATH"] = os.pathsep.join([str(bin_dir), str(cuda / "bin"), result.get("PATH", os.defpath)])
    result["CUDA_HOME"] = str(cuda)
    for name in ("ninja", "nvcc", "c++"):
        executable = shutil.which(name, path=result["PATH"])
        if not executable:
            raise RuntimeError("WISE runtime executable unavailable: " + name)
        subprocess.run([executable, "--version"], env=result, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, timeout=30)
    return result


def run(task: dict) -> None:
    service = task["judge"].get("service")
    if not service:
        score(task, clean_env())
        return

    dest = Path(task["evaluation_dir"])
    env = prepare_vllm_env(clean_env(), bin_dir=Path(service["python"]).parent)
    env.update(VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=str(dest / "judge-cache"),
               TRITON_CACHE_DIR=str(dest / "triton-cache"))
    host = service.get("host", "127.0.0.1")
    port = int(service.get("port", 18010))
    command = [
        str(service["python"]), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(service["model_path"]),
        "--served-model-name", str(task["judge"]["model"]),
        "--host", host, "--port", str(port),
        "--tensor-parallel-size", str(service.get("tensor_parallel_size", 1)),
        "--dtype", "bfloat16", "--max-model-len", str(service.get("max_model_len", 4096)),
        "--max-num-seqs", str(service.get("max_num_seqs", 1)),
        "--gpu-memory-utilization", str(service.get("gpu_memory_utilization", 0.95)),
        "--enforce-eager",
    ]
    log = dest / "metadata" / "judge.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    opener = build_opener(ProxyHandler({}))
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError("WISE judge exited before readiness")
                try:
                    with opener.open(f"http://{host}:{port}/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(10)
            else:
                raise TimeoutError("WISE judge readiness timed out")
            score(task, env)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--task":
        task_main("wise", run)
    else:
        raise SystemExit(main())
