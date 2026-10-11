import json,hashlib,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/p03'))
from common import environment,provenance,save,sha,Blocked

def now():return datetime.now(timezone.utc).isoformat()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def jsonl(path,data):
    Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
def index():
    lines=['# Experiment index','','| Run | Stage | Status | Updates |','|---|---|---|---:|']
    for run in sorted((ROOT/'runs').glob('P0*')):
        if not (run/'metrics.json').exists() or not (run/'config_resolved.json').exists():continue
        m=json.loads((run/'metrics.json').read_text());c=json.loads((run/'config_resolved.json').read_text())
        lines.append(f"| [{run.name}](../../runs/{run.name}/experiment.md) | {c.get('stage')} | {m.get('status')} | {m.get('optimizer_steps',0)} |")
    (ROOT/'docs/experiments/INDEX.md').write_text('\n'.join(lines)+'\n')
