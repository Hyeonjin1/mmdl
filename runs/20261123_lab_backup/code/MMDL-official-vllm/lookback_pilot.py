"""Prepare, smoke-test and resume the fixed MMMU Lookback pilot."""
import argparse
import ast
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from collections import Counter

from checkpointing import append_checkpoint, atomic_json, contract_hash
from lookback import (ANSWER_SIGNALS, BUDGET, INSERTIONS, MAX_LOOKBACKS, PAUSES,
                      PHRASE_SEED, WINDOW, Controller)
from run import COMMIT, REVISION, ROOT, SETTINGS, UPSTREAM
from repetition_guard import POLICY


REPO = ROOT.parents[1]
SOURCE = REPO / "results/validation900_cot8192_guard_full/predictions.jsonl"
JUDGMENTS = REPO / "results/validation900_rules_v1/judgments.jsonl"
SEOKMIN = "b5bd7c706208de056174719746511ef27588ff1f"
PARSER_ROOT = "experiments/parser_v2"
REMAINING = f"{PARSER_ROOT}/results/verified_parser_v2/remaining.jsonl"
SOURCE_SHA = "a2336c6a92acbb190ae6c1a40085c97db942158ea107ae1db5469451c3734e39"
JUDGMENTS_SHA = "f1ff9fb18fb2949f1e64dd53df5d2a8b37d38e21258aaf2b9a70d2bb8dff0fba"
REMAINING_SHA = "e4a2e91a2d11e9a1d987514b00c2653bf1549ad85b3440dad46faf66970c88ed"
DATASET_REVISION = "876ce5cb130f7f7e290ce4d9984357737d4db5cf"
MMMU_COMMIT = "b26bdefc7e5e007962fde8f93b2bb53c99c8d13d"
CONDITIONS = ("baseline", "lookback")
GROUPS = ("diagnostic_length", "normal")
SUPPLEMENT = "https://openaccess.thecvf.com/content/CVPR2026/supplemental/Bi_When_to_Think_CVPR_2026_supplemental.pdf"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=REPO)


def rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def by_id(records):
    result = {r["question_id"]: r for r in records}
    if len(result) != len(records):
        raise ValueError("Duplicate question IDs")
    return result


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen inputs differ from {path}")
    else:
        atomic_json(path, value)


def freeze_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != value:
        raise ValueError(f"Frozen code differs from {path}")
    if not path.exists():
        path.write_bytes(value)


def preflight(args):
    from PIL import Image

    remaining = command("git", "show", f"{SEOKMIN}:{REMAINING}")
    if (sha(SOURCE), sha(JUDGMENTS), hashlib.sha256(remaining).hexdigest()) != (
            SOURCE_SHA, JUDGMENTS_SHA, REMAINING_SHA):
        raise ValueError("Historical input digest mismatch")
    historical = by_id(rows(SOURCE.read_bytes()))
    judgments = by_id(rows(JUDGMENTS.read_bytes()))
    diagnostic = by_id(rows(remaining))
    if len(historical) != 900 or set(historical) != set(judgments) or len(diagnostic) != 40:
        raise ValueError("Historical ID sets mismatch")
    for qid, old in diagnostic.items():
        if (old["generation_finish_reason"] != "length" or old["prediction"] is not None
                or historical[qid]["finish_reason"] != "length"
                or old["gold"] != historical[qid]["annotation"]["answer"]):
            raise ValueError(f"Unexpected diagnostic row {qid}")

    groups = {"diagnostic_length": sorted(diagnostic)}
    rng = random.Random(3407)
    pool_sizes = {}
    for group, correct in (("normal_correct", True), ("normal_incorrect", False)):
        pool = sorted(qid for qid, row in historical.items()
                      if row["finish_reason"] == "stop" and row["annotation"].get("image_path")
                      and judgments[qid]["correct"] is correct and qid not in diagnostic)
        pool_sizes[group] = len(pool)
        groups[group] = rng.sample(pool, 20)
    manifest = json.loads(args.images.read_text())
    if (manifest["dataset"], manifest["revision"], manifest["split"], manifest["errors"]) != (
            "MMMU/MMMU", DATASET_REVISION, "validation", []):
        raise ValueError("Unexpected image manifest provenance")
    images = {row["id"]: row for row in manifest["items"]}
    targets = []
    messages = {}
    for group, ids in groups.items():
        for qid in ids:
            row = historical[qid]
            annotation = row["annotation"]
            paths = annotation["image_path"]
            item = images[qid]
            if (qid != annotation["index"] or annotation["answer"] != judgments[qid]["gold"]
                    or item["differences"] or len(paths) != len(item["images"]) or not paths):
                raise ValueError(f"Input identity or image count mismatch for {qid}")
            remap = {}
            for old_path, image in zip(paths, item["images"]):
                path = Path(image["path"])
                if path.name != Path(old_path).name or sha(path) != image["png_sha256"]:
                    raise ValueError(f"Image mismatch for {qid}: {path}")
                with Image.open(path) as data:
                    if (list(data.size) != image["size"]
                            or hashlib.sha256(data.convert("RGB").tobytes()).hexdigest() != image["rgb_pixels_sha256"]):
                        raise ValueError(f"Pixel mismatch for {qid}: {path}")
                remap[old_path] = str(path.resolve())
            prepared = copy.deepcopy(row["messages"])
            for part in prepared[-1]["content"]:
                if part["type"] == "image":
                    part["image"] = remap[part["image"]]
            if [part["image"] for part in prepared[-1]["content"] if part["type"] == "image"] != list(remap.values()):
                raise ValueError(f"Image order mismatch for {qid}")
            messages[qid] = prepared
            targets.append(dict(id=qid, group=group, old_finish=row["finish_reason"],
                                old_correct=judgments[qid]["correct"],
                                old_image_paths=paths, images=item["images"]))
    if len(messages) != 80 or set(images) != set(messages):
        raise ValueError("Expected exactly 80 distinct, mapped images-by-ID")
    freeze(args.output / "target_ids.json", dict(
        seed=3407, targets=targets, selection_pool_sizes=pool_sizes,
        historical_b0={"path": str(SOURCE.relative_to(REPO)), "sha256": SOURCE_SHA},
        historical_judgments={"path": str(JUDGMENTS.relative_to(REPO)), "sha256": JUDGMENTS_SHA},
        diagnostic={"commit": SEOKMIN, "path": REMAINING, "sha256": REMAINING_SHA},
        dataset={"name": manifest["dataset"], "revision": DATASET_REVISION, "split": "validation",
                 "image_manifest_path": str(args.images), "image_manifest_sha256": sha(args.images),
                 "historical_image_equivalence": manifest["historical_image_equivalence"]},
    ))
    frozen = args.output / "scoring"
    for relative in ("mc_parser.py", "frozen/answer_parser.py",
                     "frozen/numeric_rules.py", "frozen/terminal_format.py"):
        freeze_bytes(frozen / relative, command("git", "show", f"{SEOKMIN}:{PARSER_ROOT}/{relative}"))
    official_file = "mmmu/utils/eval_utils.py"
    freeze_bytes(frozen / "mmmu_open.py", command(
        "git", "-C", str(args.official_mmmu), "show", f"{MMMU_COMMIT}:{official_file}"))
    sys.path.insert(0, str(frozen))
    from mc_parser import decide
    from mmmu_open import eval_open, parse_open_response

    def judge(row, response, finish):
        a = row["annotation"]
        options = {key: a[key] for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if a.get(key) is not None}
        if options:
            extraction = decide(a["question"], options, response, finish)
            prediction = extraction["prediction"]
            return dict(prediction=prediction, correct=prediction == a["answer"],
                        extracted=prediction is not None, extraction=extraction)
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        gold = a["answer"]
        if isinstance(gold, str) and gold.startswith("["):
            gold = ast.literal_eval(gold)
        if not isinstance(gold, list):
            gold = str(gold)
        candidates = (sorted(set(parse_open_response(lines[-1])) - {""}, key=str)
                      if finish == "stop" and lines else [])
        return dict(prediction=candidates or None, correct=bool(candidates and eval_open(gold, candidates)),
                    extracted=bool(candidates), extraction={"route": "mmmu_open_last_line"})

    return targets, historical, messages, judge


def replay_diagnostic(targets, historical):
    audit = []
    for target in targets:
        if target["group"] != "diagnostic_length":
            continue
        text = historical[target["id"]]["result"]["gen_raw"].casefold()
        pause = min((text.index(p.casefold()) for p in PAUSES if p.casefold() in text), default=None)
        answer = min((text.index(s.casefold()) for s in ANSWER_SIGNALS if s.casefold() in text), default=None)
        audit.append(dict(id=target["id"], first_pause_char=pause, first_answer_signal_char=answer,
                          pause_before_answer=pause is not None and (answer is None or pause < answer)))
    return dict(any_pause=sum(r["first_pause_char"] is not None for r in audit),
                before_answer=sum(r["pause_before_answer"] for r in audit), rows=audit)


def load_records(path):
    # A process killed during an append can leave a partial final line.
    if not path.exists():
        path.touch()
    with path.open("r+b") as handle:
        raw = handle.read()
        if raw and not raw.endswith(b"\n"):
            end = raw.rfind(b"\n") + 1
            handle.truncate(end)
            handle.flush()
            os.fsync(handle.fileno())
            raw = raw[:end]
    return by_id(rows(raw))


def write_jsonl(path, records):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def outcome(row):
    if row["correct"]:
        return "correct"
    return "incorrect" if row["extracted"] else "unextracted"


def report(output, records, targets):
    metrics = {}
    paired = []
    events = []
    for target in targets:
        qid = target["id"]
        a, b = (records[c].get(qid) for c in CONDITIONS)
        if b is not None:
            events.extend(dict(question_id=qid, group=target["group"], **event) for event in b["events"])
        if a is not None and b is not None:
            if a["segments"][0]["prompt_token_sha256"] != b["segments"][0]["prompt_token_sha256"]:
                raise ValueError(f"Baseline and Lookback initial prefixes differ for {qid}")
            if any(row["segments"][0]["cached_prompt_tokens"] != 0 for row in (a, b)):
                raise ValueError(f"Initial request reused KV blocks for {qid}")
            fields = ("prediction", "correct", "extracted", "finish_reason", "generated_tokens",
                      "insertion_tokens", "total_tokens", "elapsed_seconds", "interventions",
                      "first_answer_phase", "first_blocked_pause")
            paired.append(dict(question_id=qid, group=target["group"], historical=target["old_correct"],
                               gold=a["gold"], baseline={k: a[k] for k in fields},
                               lookback={k: b[k] for k in fields},
                               transition=f"{outcome(a)}->{outcome(b)}"))
    for group in GROUPS:
        ids = [t["id"] for t in targets if t["group"].startswith(group)]
        group_metrics = dict(expected=len(ids))
        for condition in CONDITIONS:
            selected = [records[condition][qid] for qid in ids if qid in records[condition]]
            count = len(selected)
            totals = {field: sum(r[field] for r in selected) for field in
                      ("generated_tokens", "insertion_tokens", "total_tokens", "elapsed_seconds", "interventions")}
            restarts = [s for r in selected for s in r["segments"][1:]]
            hits = sum(r["correct"] for r in selected)
            extracted = sum(r["extracted"] for r in selected)
            group_metrics[condition] = dict(
                completed=count, correct=hits, extracted=extracted,
                accuracy=hits/count if count else None, extraction_rate=extracted/count if count else None,
                finish_reasons=dict(Counter(r["finish_reason"] for r in selected)),
                length_count=sum(r["finish_reason"] == "length" for r in selected),
                totals=totals, means={k:v/count for k,v in totals.items()} if count else None,
                intervened_questions=sum(r["interventions"] > 0 for r in selected),
                first_pause_seen=sum(r["first_pause"] is not None for r in selected),
                first_answer_phase_seen=sum(r["first_answer_phase"] is not None for r in selected),
                blocked_pause_seen=sum(r["first_blocked_pause"] is not None for r in selected),
                resubmitted_prompt_tokens=sum(s["prompt_tokens"] for s in restarts),
                cached_resubmitted_prompt_tokens=sum(s["cached_prompt_tokens"] or 0 for s in restarts),
                resume_cache_counts_available=sum(s["cached_prompt_tokens"] is not None for s in restarts),
                resubmission_first_output_seconds=sum(s["first_output_seconds"] for s in restarts),
                unaccepted_streamed_tokens=sum(s["unaccepted_streamed_tokens"] for r in selected for s in r["segments"]),
            )
        transitions = Counter(row["transition"] for row in paired if row["question_id"] in ids)
        group_metrics.update(pairs=sum(transitions.values()), transitions=dict(transitions),
                             improved=sum(transitions[f"{old}->correct"] for old in ("incorrect", "unextracted")),
                             regressed=sum(transitions[f"correct->{new}"] for new in ("incorrect", "unextracted")))
        metrics[group] = group_metrics
    write_jsonl(output / "paired_results.jsonl", paired)
    write_jsonl(output / "lookback_events.jsonl", events)
    summary = dict(expected_per_condition=len(targets), pairs=len(paired), complete=len(paired) == len(targets),
                   groups=metrics)
    atomic_json(output / "summary.json", summary)
    return summary


def validate_smoke(records, targets):
    for target in targets:
        qid = target["id"]
        a, b = (records[c][qid] for c in CONDITIONS)
        assert a["interventions"] == 0 and a["insertion_tokens"] == 0
        if not b["events"]:
            assert a["result"] == b["result"] and a["finish_reason"] == b["finish_reason"]
        else:
            assert a["first_pause"] == b["first_pause"]
            assert b["events"][0]["generated_position"] == b["first_pause"]["generated_position"]
            assert all(s["resumed_prefix_verified"] for s in b["segments"][1:])
        for row in (a, b):
            assert row["result"]["gen_raw"]
            assert row["generated_tokens"] + row["insertion_tokens"] == row["total_tokens"] <= BUDGET
            assert row["generated_tokens"] == sum(s["accepted_generated_tokens"] for s in row["segments"])
            assert len(row["segments"]) == row["interventions"] + 1 <= MAX_LOOKBACKS + 1
            assert sum(e["insertion_tokens"] for e in row["events"]) == row["insertion_tokens"]
            answer = row["first_answer_phase"]
            assert all(e["generated_since_last_trigger"] > WINDOW for e in row["events"][1:])
            assert answer is None or all(e["generated_position"] < answer["generated_position"] for e in row["events"])
    assert any(row["events"] for row in records["lookback"].values()), "Smoke did not exercise a resume"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "run", "report"))
    for name in ("output", "images", "model", "official-mmmu"):
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot(args)


def pilot(args):
    targets, historical, messages, judge = preflight(args)
    replay = replay_diagnostic(targets, historical)
    freeze(args.output / "historical_replay.json", replay)
    if args.model.name != REVISION or not (args.model / "config.json").is_file():
        raise ValueError("Select the pinned local Qwen snapshot")
    engine_args = dict(model=str(args.model), seed=3407, tensor_parallel_size=1,
                       gpu_memory_utilization=0.9, max_model_len=16384,
                       limit_mm_per_prompt={"image": 10}, trust_remote_code=True,
                       max_num_seqs=1, enable_prefix_caching=True)
    config_path = args.output / "pilot_config.json"
    cached_config = json.loads(config_path.read_text()) if config_path.exists() else None
    model_hashes = (cached_config["model_files_sha256"] if cached_config else
                    {p.name: sha(p) for p in sorted(args.model.iterdir()) if p.is_file()})
    if any(not (args.model / name).is_file() for name in model_hashes):
        raise ValueError("Pinned model snapshot files are missing")
    if not all((args.model / name).resolve().name == digest
               for name, digest in model_hashes.items() if name.endswith(".safetensors")):
        raise ValueError("Cached weight blob IDs differ from SHA-256")
    config = dict(
        input_sha256=sha(args.output / "target_ids.json"), upstream_qwen_commit=COMMIT,
        model_revision=REVISION, model_files_sha256=model_hashes, engine=engine_args,
        environment={p: importlib.metadata.version(p) for p in
                     ("vllm", "torch", "transformers", "qwen-vl-utils", "pillow")},
        generation=dict(SETTINGS, max_tokens=BUDGET), repetition_policy=POLICY,
        prefix_cache_scope="isolated by question and condition; reused within a continuation",
        scoring=dict(mc="parser_v2 deterministic final line, no cached proposal",
                     open="official MMMU open parser/scorer on last nonempty line", incomplete="unextracted",
                     commit=SEOKMIN, official_mmmu_commit=MMMU_COMMIT),
        lookback=dict(window=WINDOW, max_lookbacks=MAX_LOOKBACKS, cooldown="generated tokens > 50",
                      pause_phrases=list(PAUSES), insertion_phrases=list(INSERTIONS),
                      answer_phrases=list(ANSWER_SIGNALS), matching="casefolded substring",
                      phrase_rng_seed=PHRASE_SEED, phrase_rng_scope="per question", total_budget=BUDGET),
        code_sha256={str(p.relative_to(REPO) if p.is_relative_to(ROOT) else p.relative_to(args.output)): sha(p)
                     for p in [*(ROOT / n for n in ("lookback.py", "lookback_pilot.py", "run.py",
                                                   "streaming_guard.py", "repetition_guard.py", "checkpointing.py")),
                               *UPSTREAM.glob("*.py"), *(args.output / "scoring").rglob("*.py")]},
    )
    freeze(config_path, config)
    with (args.output / "commands.log").open("a", encoding="utf-8") as handle:
        handle.write(shlex.join([sys.executable, *sys.argv]) + "\n")
    if args.stage == "smoke":
        diag = next(row["id"] for row in replay["rows"] if row["pause_before_answer"])
        normal = next(t["id"] for t in targets if t["group"] == "normal_correct")
        targets = [t for t in targets if t["id"] in {diag, normal}]
    elif args.stage == "run" and not (args.output / "smoke" / "passed.json").is_file():
        raise RuntimeError("Smoke test must pass before starting the pilot")
    digest = contract_hash(config)
    if args.stage == "run" and json.loads((args.output / "smoke" / "passed.json").read_text())["contract"] != digest:
        raise RuntimeError("Smoke contract differs from full-run contract")
    output = args.output / "smoke" if args.stage == "smoke" else args.output
    paths = {c: output / c / "predictions.jsonl" for c in CONDITIONS}
    records = {}
    ids = {t["id"] for t in targets}
    for condition, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        records[condition] = load_records(path)
        if any(qid not in ids or r["contract"] != digest or r["condition"] != condition
               for qid, r in records[condition].items()):
            raise ValueError(f"Checkpoint contract mismatch: {path}")
    summary = report(output, records, targets)
    if args.stage in ("prepare", "report"):
        print(f"Historical diagnostic pause counts: {replay['any_pause']}/{replay['before_answer']}; {summary['pairs']} pairs saved")
        return
    if not summary["complete"]:
        infer(args, engine_args, targets, historical, messages, judge, output, paths, records, digest)
    if args.stage == "smoke":
        validate_smoke(records, targets)
        freeze(output / "passed.json", {"contract": digest, "ids": sorted(ids)})


def infer(args, engine_args, targets, historical, messages, judge, output, paths, records, digest):
    from transformers import AutoProcessor
    from vllm import SamplingParams
    from streaming_guard import StreamingGuardEngine
    sys.path.insert(0, str(UPSTREAM))
    from run_mmmu import prepare_inputs_for_vllm

    gpu = command("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                  "--format=csv,noheader").decode().strip()
    freeze(args.output / "gpu.json", {"nvidia_smi": gpu})
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    params = {k: SETTINGS[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty")}
    scheduled = [t for bucket in zip(targets[:40], targets[40:]) for t in bucket] if len(targets) == 80 else targets
    engine = StreamingGuardEngine(engine_args)
    try:
        for index, target in enumerate(scheduled):
            qid = target["id"]
            order = CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]
            for condition in order:
                if qid in records[condition]:
                    continue
                print(f"START {condition} {qid}", flush=True)
                started = time.perf_counter()
                prepared = prepare_inputs_for_vllm(messages[qid], processor)
                prepared["cache_salt"] = f"{qid}/{condition}"
                controller = Controller(processor.tokenizer, condition == "lookback")
                text, finish = controller.generate(
                    engine, prepared,
                    lambda remaining: SamplingParams(**params, max_tokens=remaining, seed=3407, stop_token_ids=[]))
                response = text.split("</think>")[-1].strip() if finish != "repetition_abort" else ""
                elapsed = time.perf_counter() - started
                record = dict(
                    question_id=qid, group=target["group"], condition=condition, contract=digest,
                    request_seed=3407, phrase_rng_seed=PHRASE_SEED, gold=historical[qid]["annotation"]["answer"],
                    result=dict(gen=response, gen_raw=text), finish_reason=finish,
                    generated_tokens=controller.generated, insertion_tokens=controller.inserted,
                    total_tokens=len(controller.trace), elapsed_seconds=elapsed,
                    interventions=len(controller.events), events=controller.events, segments=controller.segments,
                    first_pause=controller.first_pause, first_answer_phase=controller.answer_phase,
                    first_blocked_pause=controller.first_blocked_pause, repetition_evidence=controller.evidence,
                    **judge(historical[qid], response, finish),
                )
                with paths[condition].open("a", encoding="utf-8") as handle:
                    append_checkpoint(handle, record)
                records[condition][qid] = record
                report(output, records, targets)
                print(f"SAVED {condition} {qid} {finish} tokens={len(controller.trace)} "
                      f"lookbacks={len(controller.events)} seconds={elapsed:.2f}", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
