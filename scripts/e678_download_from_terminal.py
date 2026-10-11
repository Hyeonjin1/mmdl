"""Run from a network-enabled terminal: raw downloads and CPU integrity only.

No package installation, training, rendering, or dataset conversion. Archives
remain preserved; extraction and image/annotation checks are separate work.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

from e678_raw_download_probe import RAW, SOURCES


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate(path, filename):
    if filename.endswith('.json'):
        json.loads(path.read_text())
    elif filename.endswith('.zip'):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError('ZIP CRC failed: ' + bad)
    elif filename.endswith('.tgz'):
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        while stream.read(1024 * 1024):
                            pass
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    failures = 0
    for name, spec in SOURCES.items():
        folder = RAW / name
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / 'terminal_download_manifest.json'
        manifest = json.loads(output.read_text()) if output.exists() else {
            'dataset': name, 'absolute_path': str(folder),
            'documentation': spec['documentation'], 'license': spec['license'],
            'requested_version': spec['requested_version'], 'files': []}
        for filename, url, _ in spec['requests']:
            target = folder / filename
            entry = next((f for f in manifest['files'] if f['filename'] == filename), None)
            if entry is None:
                entry = {'filename': filename, 'url': url, 'actual_revision': None,
                         'downloaded_at_utc': None, 'sha256': None}
                manifest['files'].append(entry)
            entry['attempted_at_utc'] = now()
            partial = target.with_name(filename + '.partial')
            try:
                reused = target.exists()
                if not reused:
                    # Keep 8 GiB free for ongoing jobs; extraction is not started.
                    budget = shutil.disk_usage(folder).free - 8 * 1024**3
                    if budget <= 0:
                        raise RuntimeError('Insufficient space: preserve 8 GiB reserve')
                    print(name, filename, 'downloading with resume', flush=True)
                    with (folder / (filename + '.download.log')).open('a') as log:
                        result = subprocess.run([
                            'curl', '--fail', '--location', '--continue-at', '-',
                            '--retry', '2', '--connect-timeout', '20',
                            '--max-filesize', str(budget), '--output', str(partial), url],
                            stdout=log, stderr=log)
                    if result.returncode:
                        raise RuntimeError(f'curl exit {result.returncode}; see download.log')
                candidate = target if reused else partial
                digest = validate(candidate, filename)
                if not reused:
                    partial.rename(target)
                    entry['downloaded_at_utc'] = now()
                entry.update(status='reused_integrity_verified' if reused else 'downloaded_integrity_verified',
                             size_bytes=target.stat().st_size, sha256=digest, error=None)
                print(name, filename, entry['status'], flush=True)
            except Exception as error:
                failures += 1
                entry.update(status='failed', error=str(error),
                             partial_size_bytes=partial.stat().st_size if partial.exists() else 0)
                print(name, filename, 'FAILED:', error, flush=True)
            manifest.update(updated_at_utc=now(), dataset_status='partial_or_failed_pending_full_audit',
                            extraction='not_started', image_annotation_validation='not_started')
            output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
