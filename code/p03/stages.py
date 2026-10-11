"""P03 stage dispatch. GPU stages never fall back to CPU training."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import urllib.request
import zipfile
from common import ROOT, Blocked, read_manifest, require_gpu, require_records, save, sha

def inventory(cfg,run):
    words = ["chartqa","train_human","unigeo","scienceqa","ai2d","pathvqa","finqa",
             "sheetmusic","molecular"]
    roots = ["/home/user","/mnt","/media","/data","/workspace"]
    hits,errors,archives = [],[],[]
    skip = {"miniconda3","site-packages","node_modules",".git",".codex",".ssh",".aws",
            "venvs","envs",".venv",".venv-audit",".venv-p03","review_snapshots","runs",
            "pkgs","blobs","__pycache__","pip"}
    for root in roots:
        if not Path(root).exists():
            errors.append(dict(path=root,reason="not present")); continue
        def err(e): errors.append(dict(path=e.filename,reason=type(e).__name__))
        for directory,dirs,files in os.walk(root,onerror=err):
            dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".venv")]
            for name in dirs+files:
                if any(w in name.lower() for w in words):
                    p=Path(directory)/name
                    hits.append(dict(path=str(p),type="directory" if p.is_dir() else "file",
                                     bytes=p.stat().st_size if p.is_file() else None))
            for name in files:
                if name.lower().endswith(".zip"):
                    p=Path(directory)/name
                    try:
                        with zipfile.ZipFile(p) as z:
                            matched=[n for n in z.namelist() if any(w in n.lower() for w in words)]
                        if matched: archives.append(dict(path=str(p),matching_members=matched[:20]))
                    except (OSError,zipfile.BadZipFile): pass
    dns={}
    for host in ["github.com","huggingface.co","pypi.org"]:
        try: dns[host]=socket.gethostbyname(host)
        except OSError as e: dns[host]=repr(e)
    result=dict(searched_roots=roots,skipped_directory_names=sorted(skip),
                candidates=hits,archive_candidates=archives,errors=errors,dns=dns,
                disk_free_bytes=shutil.disk_usage(ROOT).free,
                available_counts_not_verified=True,
                windows_inventory_note="User supplied Windows candidate counts; not local acquisition evidence")
    save(run/"data_inventory.json",result)
    save(run/"data_manifest.json",dict(source="local inventory",records=hits,
         counts={"train":0,"check":0},processing_version="p03_inventory_v1",
         status="ChartQA usable images not found" if not archives and not hits else "inspect candidates"))
    print(json.dumps(result,indent=2))
    return result

def acquire(cfg,run):
    # Minimal range: annotation + only selected 80 images. Official GitHub first release.
    from PIL import Image
    from urllib.error import URLError
    from urllib.parse import quote
    raw=Path(cfg["raw_dir"])
    revision="044eabfc306abfe9340c5741f0093aefc5973d06"
    if (raw/"train_human.json").is_file():
        source_path=raw/"provenance.json"
        if not source_path.is_file(): raise Blocked("Existing ChartQA requires source inspection; will not overwrite")
        source=json.loads(source_path.read_text())
        images=source.get("image_records",[])
        if len(images)!=80: raise Blocked("Existing partial raw ChartQA; inspect recorded files before retry")
        for row in images:
            path=raw/"png"/row["name"]
            if not path.is_file() or sha(path)!=row["image_sha256"]: raise Blocked("Existing image hash mismatch")
            with Image.open(path) as im: im.verify()
        save(run/"data_manifest.json",dict(source=source,source_split="train_human",
             counts={"raw_images":80,"annotation_rows":7398},records=images,processing_version="p03_acquire_v1"))
        return dict(raw_images_verified=80,downloaded_new_images=0,reused_existing_raw=True)
    url="https://raw.githubusercontent.com/vis-nlp/ChartQA/"+revision+"/ChartQA%20Dataset/train/train_human.json"
    if shutil.disk_usage(ROOT).free<1*2**30: raise Blocked("Require at least1GiB free before minimal download")
    def fetch(url):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"MMDL-P03"}),timeout=15) as response:
                return response.read()
        except URLError as exc: raise Blocked("Official ChartQA network acquisition failed: "+repr(exc))
    body=fetch(url);records=json.loads(body)
    if raw.exists() and any(raw.iterdir()): raise Blocked("Raw directory is nonempty; preserve and inspect")
    raw.mkdir(parents=True,exist_ok=True);(raw/"png").mkdir()
    (raw/"train_human.json").write_bytes(body)
    names=sorted({r["imgname"] for r in records});random.Random(cfg["seed"]).shuffle(names);names=names[:80]
    source=dict(source="official vis-nlp/ChartQA GitHub first release",repository="vis-nlp/ChartQA",
         revision=revision,annotation_path="ChartQA Dataset/train/train_human.json",split="train",subset="train_human",
         download_selection_seed=cfg["seed"],annotation_rows=len(records),selected_chart_names=names,
         method="HTTPS pinned raw GitHub files",download_scope="human annotation and80 selected images",image_records=[])
    save(raw/"provenance.json",source)
    for name in names:
        if Path(name).name!=name: raise ValueError("Unsafe chart filename")
        image_url="https://raw.githubusercontent.com/vis-nlp/ChartQA/"+revision+"/"+quote("ChartQA Dataset/train/png/"+name)
        path=raw/"png"/name;path.write_bytes(fetch(image_url))
        with Image.open(path) as im: im.verify()
        source["image_records"].append(dict(name=name,path=str(path),image_sha256=sha(path),bytes=path.stat().st_size,source_url=image_url))
        save(raw/"provenance.json",source)
        save(run/"data_manifest.json",dict(source=source,source_split="train_human",
             records=source["image_records"],counts={"raw_images":len(source["image_records"]),"annotation_rows":len(records)},
             processing_version="p03_acquire_v1"))
    return dict(downloaded_images=len(names),raw_annotation_count=len(records))

def normalized_question(s):
    return re.sub(r"\s+"," ",re.sub(r"<image\s+\d+>","",s)).strip().casefold()

def prepare(cfg,run):
    from PIL import Image
    raw=Path(cfg["raw_dir"]); output=Path(cfg["processed_dir"])
    path=raw/"train_human.json"
    if not path.is_file(): raise Blocked("No ChartQA train_human annotation: "+str(path))
    provenance_path=raw/"provenance.json"
    if not provenance_path.is_file(): raise Blocked("Raw source provenance missing")
    source=json.loads(provenance_path.read_text())
    records=json.loads(path.read_text())
    evaluation_path=ROOT/"results/full900_v3/predictions.jsonl"
    evaluation_questions=set()
    if evaluation_path.is_file():
        for line in evaluation_path.read_text().splitlines():
            r=json.loads(line)
            # Source old prompt's question is before the Choices or instruction section.
            text=r["prompt"].split("\n\nChoices:\n")[0]
            text=text.split("\n\nSolve the problem.")[0]
            evaluation_questions.add(normalized_question(text))
    groups={}
    rejected=[]
    for i,r in enumerate(records):
        name=r.get("imgname")
        if not name or Path(name).name!=name: raise ValueError("Unsafe image path")
        image=raw/"png"/name
        if not image.is_file(): continue
        if not isinstance(r.get("query"),str) or not str(r.get("label","")).strip(): continue
        if normalized_question(r["query"]) in evaluation_questions:
            rejected.append(dict(index=i,reason="exact normalized evaluation question candidate")); continue
        try:
            with Image.open(image) as im: im.verify()
            with Image.open(image) as im:
                rgb=im.convert("RGB")
                pixels_hash=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                size=list(im.size)
        except Exception as e:
            rejected.append(dict(index=i,reason=repr(e))); continue
        # Same image pixels always share a group, including renamed copies.
        groups.setdefault(pixels_hash,[]).append(dict(
            original_id=source["revision"]+":train/train_human.json#"+str(i),
            original_index=i,source_split="train_human",chart_name=name,
            question=r["query"],answer=str(r["label"]).strip(),image=str(image.resolve()),
            image_sha256=sha(image),image_pixels_sha256=pixels_hash,image_size=size,
            group_id=pixels_hash))
    rng=random.Random(cfg["seed"])
    keys=sorted(groups);rng.shuffle(keys)
    if len(keys)<80:
        result=dict(valid_unique_charts=len(keys),required_unique_charts=80,
                    reason="insufficient verified distinct chart pixels; no replication")
        save(run/"data_manifest.json",dict(source=source,counts={"train":0,"check":0},
             processing_version="p03_v1",rejected=rejected,availability=result))
        raise Blocked(json.dumps(result))
    chosen=[rng.choice(sorted(groups[k],key=lambda r:r["original_index"])) for k in keys[:80]]
    train,check=chosen[:64],chosen[64:]
    assert not {r["group_id"] for r in train}&{r["group_id"] for r in check}
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in [("train",train),("check",check)]:
        with (output/(name+".jsonl")).open("x") as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    manifest=dict(source=source,source_split="train_human",seed=cfg["seed"],
        counts={"train":64,"check":16},processing_version="p03_v1",
        available_annotation_rows=len(records),valid_unique_charts=len(keys),
        record_grouping="decoded RGB pixels hash; one QA per chart",
        train=train,check=check,rejected=rejected,
        original_annotation_sha256=sha(path),
        processed_files_sha256={name:sha(output/name) for name in ["train.jsonl","check.jsonl"]},
        evaluation_overlap_check=dict(question_source=str(evaluation_path),
            questions_checked=len(evaluation_questions),exact_normalized_questions=True,
            image_hash_comparison=False,reason="MMMU original images not accessible in local data",
            mmmu_pro_checked=False,semantic_overlap_checked=False,
            limitations="Exact question candidates only; no image hashes, perceptual hashes or Pro dataset available"),
        user_prompt_policy="Only question and image, no gold/solution",
        assistant_target_policy="Original annotation concise answer plus assistant end token",
        ready_for_training=True)
    save(output/"data_manifest.json",manifest);save(run/"data_manifest.json",manifest)
    return dict(train_count=64,check_count=16,annotation_count=len(records),
                valid_unique_charts=len(keys),processed_dir=str(output))

def execute(cfg,run):
    require_records(run,cfg)
    if cfg["stage"]=="completion_audit":
        from completion_audit import audit
        return audit(cfg,run)
    if cfg["stage"]=="gpu_cleanup":
        from gpu_cleanup import cleanup
        return cleanup(cfg,run)
    if cfg["stage"]=="diagnose":
        from oom_diagnosis import diagnose
        return diagnose(cfg,run)
    if cfg["stage"]=="inventory": return inventory(cfg,run)
    if cfg["stage"]=="acquire": return acquire(cfg,run)
    if cfg["stage"]=="prepare": return prepare(cfg,run)
    if cfg["stage"]=="selftest":
        from selftest import execute_tests
        return execute_tests(cfg,run)
    from learning import structure, validate, train, evaluate
    return {"structure":structure,"validate":validate,"train":train,"evaluate":evaluate}[cfg["stage"]](cfg,run)
