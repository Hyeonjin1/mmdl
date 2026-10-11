"""Read-only GPU admission. Never terminate or modify another process."""
import csv
import io
import os
import subprocess
from common import now

def query(args):
    try:
        p = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=15)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(returncode=-1, stdout="", stderr=repr(e))

def snapshot():
    return dict(recorded_utc=now(),
        devices=query(["--query-gpu=index,uuid,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]),
        processes=query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]))

def select_gpu():
    result = snapshot()
    restriction = os.environ.get("CUDA_VISIBLE_DEVICES")
    result.update(previous_cuda_visible_devices=restriction, minimum_free_mib=18*1024,
                  policy="RTX4090, at least18GiB free, no reported compute process; this does not prove training fit")
    if result["devices"]["returncode"] or result["processes"]["returncode"]:
        result["error"] = "Cannot inspect current GPU availability via nvidia-smi; no device selected"
        return result
    occupied = {r[0].strip() for r in csv.reader(io.StringIO(result["processes"]["stdout"])) if len(r)==4}
    allowed = None if restriction is None else [x.strip() for x in restriction.split(",")]
    candidates=[]
    for row in csv.reader(io.StringIO(result["devices"]["stdout"])):
        if len(row)!=6: continue
        idx,uuid,name,total,used,free = [x.strip() for x in row]
        permitted = allowed is None or idx in allowed or any(uuid.startswith(x) for x in allowed if x.startswith("GPU-"))
        if permitted and "RTX 4090" in name and uuid not in occupied and float(free)>=18*1024:
            candidates.append(dict(physical_index=int(idx),uuid=uuid,name=name,total_mib=float(total),used_mib=float(used),free_mib=float(free)))
    if not candidates:
        result["error"]="No eligible idle RTX4090 within CUDA_VISIBLE_DEVICES restriction; no process was stopped"
        return result
    selected=max(candidates,key=lambda x:x["free_mib"])
    os.environ["CUDA_VISIBLE_DEVICES"]=selected["uuid"]
    result.update(selected=selected,logical_device=0,actual_cuda_visible_devices=selected["uuid"])
    return result
