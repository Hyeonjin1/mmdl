"""Run records and fail-closed guards shared by every P03 stage."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("experiment.md", "config_resolved.json", "environment.json",
            "data_manifest.json", "commands.sh", "metrics.json", "code_provenance.json")

class Blocked(RuntimeError):
    pass

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)

def now():
    return datetime.now(timezone.utc).isoformat()

def environment():
    packages = {}
    for name in ["torch", "torchvision", "transformers", "peft", "accelerate", "pillow", "numpy", "huggingface_hub", "safetensors", "bitsandbytes"]:
        try: packages[name] = md.version(name)
        except md.PackageNotFoundError: packages[name] = None
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  packages=packages, recorded_utc=now())
    p = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    result["nvidia_smi"] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    try:
        import torch
        result.update(torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                      visible_gpu_count=torch.cuda.device_count())
        result["gpus"] = [
            dict(index=i, name=torch.cuda.get_device_name(i),
                 vram_bytes=torch.cuda.get_device_properties(i).total_memory,
                 free_total_bytes=list(torch.cuda.mem_get_info(i)))
            for i in range(torch.cuda.device_count())]
    except Exception as e:
        result["cuda_probe_error"] = repr(e)
    return result

def provenance():
    # Snapshot all local source/config/docs files, including uncommitted existing work.
    files = {}
    for folder in ["code", "scripts", "configs", "docs", "assignment", "reports"]:
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in [".py", ".json", ".sh", ".md", ".txt"]:
                files[str(p.relative_to(ROOT))] = sha(p)
    for name in ["README.md", "RUNBOOK.md", ".gitignore"]:
        if (ROOT/name).is_file(): files[name] = sha(ROOT/name)
    p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    is_repo = p.returncode == 0 and p.stdout.strip() == str(ROOT)
    git_head = None
    if is_repo:
        git_head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return dict(repository="eses0070/MMDL",
                upstream_snapshot_commit="0c0b415f6f93923a4249d3da32c5cbb447a518ea",
                local_git_repository=is_repo, local_git_head=git_head,
                local_source_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
                local_file_sha256=files, local_changes_note="Local snapshot with added files; original files preserved. Hash inventory replaces git diff when no .git exists.")

def index():
    target = ROOT / "docs/experiments/INDEX.md"
    lines = ["# Experiment index", "", "| Run | Stage | Status | Optimizer steps | Record |",
             "|---|---|---|---:|---|"]
    for run in sorted((ROOT/"runs").glob("P03_*")):
        if not (run/"metrics.json").is_file(): continue
        m = json.loads((run/"metrics.json").read_text())
        c = json.loads((run/"config_resolved.json").read_text())
        lines.append(f"| {run.name} | {c['stage']} | {m['status']} | {m.get('optimizer_steps', 0)} | [experiment](../../runs/{run.name}/experiment.md) |")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines)+"\n")

def require_records(run, cfg):
    missing = [name for name in REQUIRED if not (run/name).is_file()]
    if missing: raise Blocked("Missing mandatory records: " + ", ".join(missing))
    if not (run/"logs").is_dir(): raise Blocked("Missing logs directory")
    if json.loads((run/"config_resolved.json").read_text()) != cfg:
        raise Blocked("Configuration record does not match execution")
    if os.environ.get("P03_RUN_DIR") != str(run):
        raise Blocked("Stage must execute through common launcher")

def require_gpu(cfg):
    import torch
    if int(os.environ.get("WORLD_SIZE","1")) != 1: raise Blocked("P03 requires world size 1")
    if not torch.cuda.is_available(): raise Blocked("No accessible CUDA GPU; nvidia-smi fails in this session")
    if not torch.cuda.is_bf16_supported(): raise Blocked("BF16 not supported")
    if "RTX 4090" not in torch.cuda.get_device_name(0): raise Blocked("P03 requires one RTX 4090; observed " + torch.cuda.get_device_name(0))
    free,total = torch.cuda.mem_get_info(0)
    if free < 18 * 2**30: raise Blocked("Selected GPU has less than 18GiB free; admission threshold, not a guarantee that BF16 training fits")
    if total < 22 * 2**30: raise Blocked("Selected GPU does not expose approximately 24GB VRAM")
    torch.ones(1, device="cuda").add_(1)
    return dict(selected_gpu=0, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), name=torch.cuda.get_device_name(0), free_bytes=free, total_bytes=total)

def read_manifest(cfg, run):
    path = Path(cfg["processed_dir"]) / "data_manifest.json"
    if not path.is_file(): raise Blocked("Processed ChartQA manifest is absent: " + str(path))
    manifest = json.loads(path.read_text())
    if manifest.get("source_split") != "train_human" or manifest.get("counts") != {"train":64,"check":16}:
        raise Blocked("Need exactly 64 training /16 disjoint chart checks from train_human")
    raw_annotation=Path(cfg["raw_dir"])/"train_human.json"
    if not raw_annotation.is_file() or sha(raw_annotation)!=manifest["original_annotation_sha256"]:
        raise Blocked("Original ChartQA annotation changed or missing")
    annotations=json.loads(raw_annotation.read_text())
    for name,expected in manifest["processed_files_sha256"].items():
        if sha(Path(cfg["processed_dir"])/name)!=expected:
            raise Blocked("Processed file changed: "+name)
    for row in manifest["train"]+manifest["check"]:
        original=annotations[row["original_index"]]
        if (row["question"]!=original["query"] or row["answer"]!=str(original["label"]).strip()
                or row["chart_name"]!=original["imgname"]):
            raise Blocked("Question/answer/image does not match original annotation")
    save(run/"data_manifest.json", manifest)
    return manifest
