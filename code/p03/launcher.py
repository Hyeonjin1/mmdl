"""Create mandatory run records before executing; preserve every prior run."""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import time
import traceback
from common import ROOT, Blocked, REQUIRED, environment, index, now, provenance, require_records, save, sha

FIELDS = ("purpose", "model_id", "model_revision", "seed", "raw_dir", "processed_dir",
          "train_size", "check_size", "world_size", "bf16", "lora_rank", "lora_alpha",
          "lora_dropout", "micro_batch", "gradient_accumulation", "optimizer_steps",
          "learning_rate", "weight_decay", "warmup_steps", "max_grad_norm",
          "max_sequence_length", "min_pixels", "max_pixels", "gradient_checkpointing",
          "packing", "assistant_only_loss", "generation_max_new_tokens")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stage", choices=["inventory", "acquire", "prepare", "structure", "validate", "train", "evaluate", "selftest", "diagnose", "gpu_cleanup", "completion_audit"], required=True)
    p.add_argument("--run-id")
    p.add_argument("--external", action="store_true", help="Record externally orchestrated connector acquisition before any file transfer")
    p.add_argument("--finish-external", choices=["completed","blocked","error"])
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    missing = [k for k in FIELDS if k not in cfg]
    if missing: p.error("Missing mandatory settings: " + ", ".join(missing))
    cfg.update(stage=args.stage, config_source=str(args.config.resolve()))
    if cfg["model_id"] != "Qwen/Qwen3-VL-4B-Instruct" or cfg["model_revision"] != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
        p.error("Wrong student or revision")
    if cfg["world_size"] != 1 or cfg["packing"] or not cfg["assistant_only_loss"]:
        p.error("Invalid world size/packing/loss policy")
    if cfg["micro_batch"]!=1 or not cfg["bf16"] or cfg.get("optimizer")!="AdamW":
        p.error("This implementation supports microbatch1, BF16 LoRA and AdamW only")
    run_id = args.run_id or "P03_" + time.strftime("%Y%m%dT%H%M%SZ",time.gmtime()) + "_" + args.stage
    if "/" in run_id or not run_id.startswith("P03_"): p.error("Invalid run ID")
    run = ROOT / "runs" / run_id
    os.environ["P03_RUN_DIR"] = str(run)
    if args.finish_external:
        require_records(run,cfg)
        metrics = json.loads((run/"metrics.json").read_text())
        if metrics["status"] != "running_external": p.error("Only an active external record can be finalized")
        metrics.update(status=args.finish_external,ended_utc=now())
        save(run/"metrics.json",metrics)
        with (run/"experiment.md").open("a") as f: f.write("\nExternal acquisition outcome: " + args.finish_external + "\nSee logs and data_manifest.json.\n")
        index()
        return
    run.mkdir(parents=True,exist_ok=False)
    (run/"logs").mkdir()
    save(run/"config_resolved.json",cfg)
    from gpu_admission import select_gpu
    if args.stage == "gpu_cleanup": os.environ["CUDA_VISIBLE_DEVICES"]="0"
    selection = select_gpu() if args.stage in ("train", "evaluate") else None
    save(run/"environment.json",environment())
    if selection is not None: save(run/"gpu_selection.json",selection)
    save(run/"code_provenance.json",provenance())
    save(run/"data_manifest.json",dict(status="not_prepared",source="ChartQA",source_split="train_human",
                                      counts={"train":0,"check":0},processing_version="p03_v1",
                                      records=[],expected_processed_dir=cfg["processed_dir"]))
    save(run/"metrics.json",dict(status="created",started_utc=now(),optimizer_steps=0))
    command = [sys.executable,str(Path(__file__).resolve()),"--config",str(args.config.resolve()),"--stage",args.stage]
    if args.external: command += ["--external"]
    (run/"commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd "+shlex.quote(str(ROOT))+"\n"+shlex.join(command)+"\n")
    (run/"experiment.md").write_text(
        "# " + run_id + "\n\n## Purpose\n" + cfg["purpose"] +
        "\n\n## Actual conditions\nStage: " + args.stage +
        "\nSee config_resolved.json, environment.json, data_manifest.json and code_provenance.json." +
        "\n\n## Results\nPending. Training is not KD; no performance improvement claim." +
        "\n\n## Limitations\n563/900 is a mixed pretraining evaluation record, not a matched scoring baseline. No full MMMU evaluation in P03."+
        "\n\n## Next steps\nVerify images, assistant mask, actual GPU updates and adapter reload before KD.\n")
    require_records(run,cfg)
    start = time.perf_counter()
    metrics = json.loads((run/"metrics.json").read_text())
    metrics["status"] = "running_external" if args.external else "running"
    save(run/"metrics.json",metrics)
    index()
    if args.external:
        print(str(run)); return
    def interrupted(signum, frame): raise KeyboardInterrupt("signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    status, exitcode, result = "completed", 0, {}
    with (run/"logs/stage.log").open("x",buffering=1) as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if selection is not None and selection.get("error"):
                    raise Blocked(selection["error"])
                result = importlib.import_module("stages").execute(cfg,run)
        except KeyboardInterrupt as exc:
            status,exitcode,result = "interrupted",130,{"error":repr(exc)}
            log.write(traceback.format_exc())
        except BaseException as exc:
            status,exitcode,result = ("blocked" if isinstance(exc,Blocked) else "error"),1,{"error":repr(exc)}
            if "out of memory" in str(exc).lower(): status="oom"
            log.write(traceback.format_exc())
            save(run/"logs/error.json",dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    progress_path = run/"progress.json"
    if progress_path.is_file(): result = {**json.loads(progress_path.read_text()),**result}
    phase_path = run/"execution_phase.json"
    if phase_path.is_file():
        phase = json.loads(phase_path.read_text())
        result.update(last_execution_phase=phase)
        result["optimizer_steps"] = max(result.get("optimizer_steps",0), phase.get("optimizer_steps",0))
        try:
            import torch
            if torch.cuda.is_initialized():
                result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except Exception:
            pass
    metrics.update(result,status=status,ended_utc=now(),wall_seconds=time.perf_counter()-start)
    save(run/"metrics.json",metrics)
    with (run/"experiment.md").open("a") as f:
        f.write("\n## Execution outcome\nStatus: "+status+"\n\n"+json.dumps(metrics,ensure_ascii=False,indent=2)+"\n")
    index()
    # Reporting runs AFTER terminal experiment records are saved. Publication
    # failures have independent records and never change experiment exit/status.
    import subprocess
    with (run/"logs/publication.log").open("a") as publication_log:
        try:
            publication = subprocess.run([sys.executable,str(ROOT/"code/reporting/reports.py"),
                "--run-id",run_id,"--publish"],stdout=publication_log,stderr=publication_log,timeout=180)
        except Exception as exc:
            save(run/"publication_hook_error.json",dict(error=repr(exc),experiment_status_unchanged=status))
    print(str(run), status)
    sys.exit(exitcode)

if __name__ == "__main__": main()
