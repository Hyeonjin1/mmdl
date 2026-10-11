"""User-authorized, PID-specific GPU0 cleanup; no blanket Python termination."""
import csv
import io
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import time
from common import Blocked, read_manifest, save
from gpu_admission import snapshot

def info(pid):
    base=Path('/proc')/str(pid)
    stat=(base/'stat').read_text().rsplit(')',1)[1].split()
    argv=(base/'cmdline').read_bytes().split(b'\0')
    command=' '.join(x.decode(errors='replace') for x in argv if x)
    command=re.sub(r'(?i)((?:token|password|secret|api[-_]?key)[=\s]+)\S+',r'\1[REDACTED]',command)
    return dict(pid=pid,ppid=int(stat[1]),start_time_ticks=stat[19],
                owner=pwd.getpwuid(base.stat().st_uid).pw_name,
                executable=os.readlink(base/'exe'),command=command,state=stat[0])

def same_alive(original):
    try:
        current=info(original['pid'])
        return current['start_time_ticks']==original['start_time_ticks'] and current['state']!='Z'
    except FileNotFoundError:
        return False

def wait_exit(original,seconds):
    end=time.monotonic()+seconds
    while same_alive(original) and time.monotonic()<end:
        time.sleep(.2)
    return not same_alive(original)

def cleanup(cfg,run):
    read_manifest(cfg,run)
    if cfg.get('gpu_cleanup_authorized') is not True:
        raise Blocked('Explicit GPU cleanup authorization missing from config')
    report=dict(before=snapshot(),targets=[],terminated_pids=[],errors=[],
                protected_policy='Keep current process/ancestors, Codex, SSH and display processes; only live GPU0 Python compute PIDs',
                historical_pid_not_a_target=1202089)
    path=run/'gpu_cleanup.json'
    save(path,report)
    before=report['before']
    if before['devices']['returncode'] or before['processes']['returncode']:
        report['errors'].append('nvidia-smi cannot enumerate live GPU0 PIDs; no signal sent')
    else:
        devices=list(csv.reader(io.StringIO(before['devices']['stdout'])))
        uuid=next((r[1].strip() for r in devices if len(r)==6 and r[0].strip()=='0'),None)
        if uuid is None:
            raise Blocked('Physical GPU0 not found')
        protected={os.getpid()}
        ancestor=os.getpid()
        while ancestor>1:
            ancestor=info(ancestor)['ppid'];protected.add(ancestor)
        rows=list(csv.reader(io.StringIO(before['processes']['stdout'])))
        for row in rows:
            if len(row)!=4 or row[0].strip()!=uuid: continue
            pid=int(row[1].strip())
            target=dict(pid=pid,reported_memory_mib=row[3].strip(),signals=[])
            report['targets'].append(target)
            try:
                original=info(pid);target['process']=original
                try: target['parent']=info(original['ppid'])
                except FileNotFoundError: target['parent']='Parent exited before inspection'
                name=Path(original['executable']).name.lower()
                if pid in protected or not name.startswith('python') or any(x in original['command'].lower() for x in ['codex','sshd','xorg']):
                    target['skipped']='Protected session/display or not a Python executable'
                    save(path,report);continue
                save(path,report)  # Identity, owner, command and parent recorded BEFORE signal.
                current=snapshot()
                if current['processes']['returncode']:
                    raise Blocked('Cannot re-confirm GPU occupancy immediately before signal')
                still_gpu=any(len(r)==4 and r[0].strip()==uuid and r[1].strip()==str(pid)
                    for r in csv.reader(io.StringIO(current['processes']['stdout'])))
                if not still_gpu or not same_alive(original):
                    target['skipped']='Original PID no longer live on GPU0';continue
                os.kill(pid,signal.SIGTERM);target['signals'].append('SIGTERM');save(path,report)
                if not wait_exit(original,10):
                    os.kill(pid,signal.SIGKILL);target['signals'].append('SIGKILL');save(path,report)
                    if not wait_exit(original,5): raise Blocked('PID remained alive after SIGKILL')
                report['terminated_pids'].append(pid)
            except Exception as exc:
                target['error']=repr(exc);report['errors'].append(dict(pid=pid,error=repr(exc)))
            finally: save(path,report)
    report['after']=snapshot()
    try:
        import torch
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        a=torch.tensor([2.,3.],device='cuda:0')
        value=float((a*a).sum().item());torch.cuda.synchronize()
        assert value==13.
        report['cuda_tensor_check']=dict(passed=True,result=value,free_total_bytes=list(torch.cuda.mem_get_info(0)))
        del a
    except Exception as exc:
        report['cuda_tensor_check']=dict(passed=False,error=repr(exc))
        report['errors'].append('CUDA tensor operation failed: '+repr(exc))
    save(path,report)
    if report['errors']: raise Blocked('GPU cleanup/probe not complete; see gpu_cleanup.json for exact errors')
    return dict(terminated_pids=report['terminated_pids'],cuda_tensor_check=report['cuda_tensor_check'])
