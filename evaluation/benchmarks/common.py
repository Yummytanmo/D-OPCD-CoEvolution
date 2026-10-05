"""Shared task, record, and input format helpers for benchmark evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2] / "D-OPCD"
EVALUATOR_ASSETS = ROOT.parent / "evaluation" / ".assets"


def project_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def evaluator_root(task: dict, benchmark: str) -> Path:
    return project_path(task.get("evaluator_root") or EVALUATOR_ASSETS / benchmark)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inside(root: Path, raw_path: str, *, base: Path | None = None) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base or root) / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact escapes Agent run: {path}")
    return path


def ensure_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Conflicting symlink: {destination}")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing evaluator input: {destination}")
    destination.symlink_to(source)


def collect_records(
    *,
    benchmark: str,
    split: str,
    manifest: Path,
    agent_run: Path,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = read_jsonl(manifest)
    provenance = read_json(agent_run / "provenance.json")
    index_rows = read_jsonl(agent_run / "index.jsonl")
    manifest_hash = digest(manifest)
    if len(sources) != expected_count:
        raise ValueError(f"Manifest count={len(sources)}, expected={expected_count}")
    if provenance.get("status") != "completed":
        raise ValueError("Agent provenance is not completed")
    if provenance.get("dataset") != benchmark or provenance.get("split") != split:
        raise ValueError("Agent provenance dataset/split mismatch")
    if (provenance.get("source_manifest") or {}).get("sha256") != manifest_hash:
        raise ValueError("Agent provenance manifest hash mismatch")
    if int(provenance.get("completed_count", 0) or 0) != expected_count:
        raise ValueError("Agent provenance completed count mismatch")
    if len(index_rows) != expected_count:
        raise ValueError("Agent index count mismatch")

    source_by_id = {str(row.get("source_sample_id") or row.get("sample_id") or ""): row for row in sources}
    if "" in source_by_id or len(source_by_id) != expected_count:
        raise ValueError("Manifest source IDs are missing or duplicated")
    index_by_id = {str(row.get("source_sample_id") or ""): row for row in index_rows}
    if set(index_by_id) != set(source_by_id):
        raise ValueError("Agent index source IDs do not exactly match the manifest")

    records: list[dict[str, Any]] = []
    for source in sources:
        sample_id = str(source.get("source_sample_id") or source.get("sample_id"))
        index = index_by_id[sample_id]
        trajectory_path = inside(agent_run, str(index["trajectory_path"]))
        trajectory = read_json(trajectory_path)
        if trajectory.get("status") != "completed":
            raise ValueError(f"Non-completed trajectory: {sample_id}")
        if trajectory.get("original_query") != str(source.get("prompt") or "").strip():
            raise ValueError(f"Source prompt mismatch: {sample_id}")
        selected_iteration = trajectory.get("selected_iteration")
        selected = [
            attempt
            for attempt in trajectory.get("attempts", [])
            if attempt.get("iteration") == selected_iteration
        ]
        if len(selected) != 1:
            raise ValueError(f"Selected iteration missing or duplicated: {sample_id}")
        image_path = inside(
            agent_run,
            str(selected[0].get("image_path") or ""),
            base=trajectory_path.parent,
        )
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"Selected image missing or empty: {sample_id}")
        image_hash = digest(image_path)
        if {
            image_hash,
            selected[0].get("image_sha256"),
            trajectory.get("selected_image_sha256"),
            index.get("selected_image_sha256"),
        } != {image_hash}:
            raise ValueError(f"Selected image hash mismatch: {sample_id}")
        records.append(
            {
                "source_sample_id": sample_id,
                "benchmark": benchmark,
                "split": split,
                "prompt": source["prompt"],
                "source": source,
                "selected_prompt": trajectory.get("selected_prompt"),
                "selected_iteration": selected_iteration,
                "selected_image_path": str(image_path),
                "selected_image_sha256": image_hash,
                "trajectory_path": str(trajectory_path),
                "trajectory_protocol_sha256": trajectory.get("protocol_sha256"),
            }
        )
    return records, provenance


def stage_geneval(records: list[dict[str, Any]], input_root: Path) -> None:
    image_root = input_root / "images"
    for index, record in enumerate(records):
        folder = image_root / f"{index:05d}"
        metadata = dict(record["source"].get("source_metadata") or {})
        metadata["prompt"] = record["prompt"]
        write_json(folder / "metadata.jsonl", metadata)
        ensure_symlink(
            Path(record["selected_image_path"]), folder / "samples" / "0000.png"
        )


def geneval2_inputs(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    benchmark_rows: list[dict[str, Any]] = []
    image_paths: dict[str, str] = {}
    for record in records:
        source = record["source"]
        metadata = dict(source.get("source_metadata") or {})
        for key in ("vqa_list", "skills", "atom_count"):
            if key in source:
                if key in metadata and metadata[key] != source[key]:
                    raise ValueError(f"Conflicting GenEval2 {key}: {record['source_sample_id']}")
                metadata[key] = source[key]
        vqa_list = metadata.get("vqa_list")
        skills = metadata.get("skills")
        atom_count = metadata.get("atom_count")
        if (not isinstance(vqa_list, list) or not vqa_list
                or any(not isinstance(pair, list) or len(pair) != 2 for pair in vqa_list)
                or not isinstance(skills, list) or len(skills) != len(vqa_list)
                or not isinstance(atom_count, int) or isinstance(atom_count, bool)
                or atom_count < 3 or atom_count > 10):
            raise ValueError(f"Incomplete GenEval2 evaluator metadata: {record['source_sample_id']}")
        benchmark_rows.append({"prompt": record["prompt"], **metadata})
        if record["prompt"] in image_paths:
            raise ValueError("GenEval2 prompt keys are not unique in this split")
        image_paths[record["prompt"]] = record["selected_image_path"]
    return benchmark_rows, image_paths


def stage_geneval2(records: list[dict[str, Any]], input_root: Path) -> None:
    benchmark_rows, image_paths = geneval2_inputs(records)
    write_jsonl(input_root / "benchmark_data.jsonl", benchmark_rows)
    write_json(input_root / "image_paths.json", image_paths)


def adapt_geneval2_source(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["source_sample_id"] = row.get("source_sample_id") or row.get("sample_id")
    metadata = dict(row.get("source_metadata") or {})
    for key in ("atom_count", "vqa_list", "skills"):
        if key not in metadata and key in row:
            metadata[key] = row[key]
    if not metadata.get("vqa_list") or len(metadata["vqa_list"]) != len(metadata.get("skills", [])):
        raise ValueError("Invalid GenEval2 VQA/skills annotations")
    row["source_metadata"] = metadata
    return row


def stage_checkpoint(task: dict[str, Any]) -> dict[str, Any]:
    """Prepare generated checkpoint images for the official evaluator."""
    benchmark = task["benchmark"]
    if benchmark not in ("geneval", "geneval2"):
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    manifest_path = Path(task["manifest"]).expanduser().resolve()
    output_dir = Path(task["evaluation_dir"]).expanduser().resolve()
    records_path = output_dir / "generation" / "records.jsonl"
    manifest = read_jsonl(manifest_path)
    if benchmark == "geneval2":
        manifest = [adapt_geneval2_source(row) for row in manifest]
    records = read_jsonl(records_path)
    expected_count = task["expected_count"]
    if len(manifest) != expected_count or len(records) != expected_count:
        raise ValueError(
            f"manifest/records count={len(manifest)}/{len(records)}, expected={expected_count}"
        )
    source_ids = [str(row.get("source_sample_id") or "") for row in manifest]
    if "" in source_ids or len(set(source_ids)) != expected_count:
        raise ValueError("Manifest source IDs are missing or duplicated")
    record_by_id = {str(row.get("sample_id") or ""): row for row in records}
    if set(record_by_id) != set(source_ids) or len(record_by_id) != expected_count:
        raise ValueError("Generated record IDs do not exactly match the manifest")

    staged = []
    for source in manifest:
        sample_id = str(source["source_sample_id"])
        record = record_by_id[sample_id]
        prompt = str(source.get("prompt") or "").strip()
        if (
            (source.get("benchmark") or source.get("dataset_id")) != benchmark
            or source.get("split") != task["split"]
            or record.get("benchmark") != benchmark
            or record.get("split") != task["split"]
            or record.get("status") != "complete"
            or record.get("original_prompt") != prompt
            or record.get("generation_prompt") != prompt
        ):
            raise ValueError(f"Manifest/generated record mismatch: {sample_id}")
        image_path = Path(str(record.get("image_path") or "")).expanduser().resolve()
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"Generated image is missing or empty: {sample_id}")
        staged.append({
            "source_sample_id": sample_id, "benchmark": benchmark,
            "split": task["split"], "prompt": prompt, "source": source,
            "selected_image_path": str(image_path), "checkpoint_step": task["step"],
            "generated_image_path": str(image_path),
            "generated_image_sha256": digest(image_path), "generation_record": record,
        })

    input_root = output_dir / "inputs"
    if benchmark == "geneval":
        stage_geneval(staged, input_root)
    else:
        stage_geneval2(staged, input_root)
    plan = {
        "schema_version": 1, "status": "completed", "benchmark": benchmark,
        "split": task["split"], "checkpoint_step": task["step"],
        "sample_count": len(staged), "images_per_prompt": 1,
        "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
        "generation_records": str(records_path), "output_dir": str(output_dir),
        "execute_requested": True,
    }
    write_jsonl(output_dir / "metadata/records.jsonl", staged)
    write_json(output_dir / "metadata/input-preparation.json", plan)
    return plan


def stage_agent_inputs(task: dict[str, Any]) -> dict[str, Any]:
    """Use the Agent-selected images as GenEval or GenEval2 evaluator input."""
    benchmark = task["benchmark"]
    if benchmark not in ("geneval", "geneval2"):
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    manifest = Path(task["manifest"]).expanduser().resolve()
    agent_run = Path(task["agent_run_dir"]).expanduser().resolve()
    output = Path(task["evaluation_dir"]).expanduser().resolve()
    records, provenance = collect_records(
        benchmark=benchmark, split=task["split"], manifest=manifest,
        agent_run=agent_run, expected_count=task["expected_count"],
    )
    if benchmark == "geneval":
        stage_geneval(records, output / "inputs")
    else:
        stage_geneval2(records, output / "inputs")
    plan = {
        "schema_version": 1, "status": "completed", "benchmark": benchmark,
        "split": task["split"], "sample_count": len(records), "images_per_prompt": 1,
        "manifest": str(manifest), "manifest_sha256": digest(manifest),
        "agent_run_dir": str(agent_run),
        "agent_run_id": provenance.get("agent_run_id"),
        "agent_protocol_sha256": provenance.get("protocol_sha256"),
        "output_dir": str(output), "execute_requested": True,
    }
    write_jsonl(output / "metadata/records.jsonl", records)
    write_json(output / "metadata/input-preparation.json", plan)
    return plan


def clean_env(*, keep_proxy: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in {"WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                   "MASTER_ADDR", "MASTER_PORT"} or (
            not keep_proxy and key.lower() in {"http_proxy", "https_proxy", "all_proxy"}
        ):
            env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    return env


def run_logged(command: list[str], *, cwd: Path, log: Path,
               env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env or clean_env(), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def run_task(path: Path, benchmark: str, worker: Callable[[dict], None]) -> None:
    task = read_json(path)
    if task.get("benchmark") != benchmark:
        raise ValueError(f"Expected {benchmark} task")
    count = task.get("expected_count")
    if not isinstance(count, int) or count < 1:
        raise ValueError("expected_count must be positive")
    dest = Path(task["evaluation_dir"])
    lifecycle = dest / "metadata" / "evaluator-lifecycle.json"
    started = datetime.now(timezone.utc).isoformat()
    status = "failed"
    try:
        worker(task)
        metrics = read_json(dest / "scores" / "metrics.json")
        if metrics.get("status") != "completed" or metrics.get("sample_count") != count:
            raise ValueError(f"Incomplete {benchmark} scores: {dest}")
        status = "completed"
    finally:
        write_json(lifecycle, {"benchmark": benchmark, "split": task.get("split"),
                               "expected_count": count, "status": status,
                               "started_at": started,
                               "finished_at": datetime.now(timezone.utc).isoformat()})


def task_main(benchmark: str, worker: Callable[[dict], None]) -> None:
    parser = argparse.ArgumentParser(description=f"Score {benchmark} images")
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    run_task(args.task, benchmark, worker)
