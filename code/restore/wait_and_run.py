"""Wait for the existing verified download, then run one recorded GPU suite."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.restore/launch_state.json'


def record(status, **details):
    value = dict(status=status, pid=os.getpid(),
                 timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 **details)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)
    print(json.dumps(value), flush=True)


def downloader_running():
    expected = str(ROOT / 'code/restore/model_parallel.py').encode()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / 'cmdline').read_bytes().split(b'\0')
            if expected in arguments:
                return True
        except (OSError, PermissionError):
            pass
    return False


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    (ROOT / '.locks').mkdir(exist_ok=True)
    with (ROOT / '.locks/restore-waiter.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists():
            previous = json.loads(STATE.read_text())
            if previous.get('status') in ['failed', 'completed']:
                raise RuntimeError('Prior launch receipt exists; deliberate review required before another launch')
        record('waiting_for_verified_download')
        while True:
            cfg = json.loads((ROOT / 'configs/restore/E123.json').read_text())
            model = Path(cfg['model_path'])
            manifest_path = model / 'model_manifest.json'
            if manifest_path.exists():
                break
            if not downloader_running():
                # Account for the final manifest/config write race.
                time.sleep(2)
                cfg = json.loads((ROOT / 'configs/restore/E123.json').read_text())
                model = Path(cfg['model_path'])
                if not (model / 'model_manifest.json').exists():
                    raise RuntimeError('Downloader exited before verified model completion; partial files preserved')
                manifest_path = model / 'model_manifest.json'
                break
            time.sleep(15)
        record('verifying_download', model_path=str(model))
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('complete') or manifest['revision'] != cfg['model_revision']:
            raise RuntimeError('Model completion/revision contract failed')
        names = {entry['name'] for entry in manifest['files']}
        for required in ['config.json', 'model.safetensors.index.json',
                         'tokenizer.json', 'preprocessor_config.json']:
            if required not in names:
                raise RuntimeError('Missing required pinned model file: ' + required)
        index = json.loads((model / 'model.safetensors.index.json').read_text())
        if not set(index['weight_map'].values()).issubset(names):
            raise RuntimeError('Weight index refers to absent shards')
        for entry in manifest['files']:
            path = model / entry['name']
            if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                raise RuntimeError('Downloaded file integrity failed: ' + entry['name'])
        record('host_cuda_probe')
        subprocess.run([cfg['train_python'], str(ROOT / 'code/restore/host_probe.py')],
                       cwd=ROOT, check=True)
        record('running_suite')
        try:
            result = subprocess.run([cfg['train_python'], str(ROOT / 'code/restore/run.py'),
                                     '--suite'], cwd=ROOT)
        finally:
            subprocess.run([cfg['train_python'], str(ROOT / 'code/restore/report.py')], cwd=ROOT)
        if result.returncode:
            raise RuntimeError('Recorded GPU suite failed; inspect suite_state.json and failed run logs')
        record('completed')


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        print('An existing waiter owns the lock; no duplicate launch.', flush=True)
        sys.exit(1)
    except BaseException as error:
        record('failed', error=repr(error), traceback=traceback.format_exc())
        raise
