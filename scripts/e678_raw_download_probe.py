"""CPU-only source download probe. No training, generation, or data conversion.

This records failures honestly. Archive requests are HEAD-only until storage for
both the archive and expanded contents has been established. JSON requests use
curl resume into .partial; completed JSON is parsed before atomic rename.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'datasets/raw'
SOURCES = {
    'ScienceQA': {
        'documentation': 'https://github.com/lupantech/ScienceQA',
        'requested_version': 'main (not resolved to commit; not a pinned revision)',
        'license': 'Dataset CC BY-NC-SA 4.0; repository code license is distinct',
        'requests': [
            ('problems.json', 'https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json', 'json'),
            ('pid_splits.json', 'https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/pid_splits.json', 'json'),
            *[(f'{s}.zip', f'https://scienceqa.s3.us-west-1.amazonaws.com/images/{s}.zip', 'archive_head') for s in ('train', 'val', 'test')],
        ],
        'official_split_documentation': 'problems.json split field; pid_splits.json requested but not acquired',
    },
    'AI2D': {
        'documentation': 'https://prior.allenai.org/projects/diagram-understanding',
        'requested_version': 'original AI2D release; object version unknown',
        'license': 'AWS registry lists CC BY-SA; exact version must be checked in acquired files',
        'requests': [('ai2d-all.zip', 'https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip', 'archive_head')],
        'source_note': 'Official AWS registry identifies ai2-public-datasets bucket. Object path corroborated by google-research/pix2struct. PRIOR archive link could not be fetched by browser. No unofficial mirror used.',
        'official_split_documentation': 'PRIOR provides Test Set IDs link; target URL and file not acquired',
    },
    'Fusion360Gallery': {
        'documentation': 'https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/reconstruction.md',
        'requested_version': 'reconstruction r1.0.1',
        'license': 'Autodesk dataset license, updated 11/2021: non-commercial research; no redistribution of entire dataset',
        'license_url': 'https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md',
        'requests': [('r1.0.1.zip', 'https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/reconstruction/r1.0.1/r1.0.1.zip', 'archive_head')],
        'official_split_documentation': 'train_test.json, 6900 train / 1725 test designs (documentation only)',
        'excluded': ['assembly', 'segmentation', 'extrude tools', 'preprocessed face extrusion sequences'],
    },
    'PrIMuS': {
        'documentation': 'https://grfia.dlsi.ua.es/primus/',
        'requested_version': 'basic PrIMuS / Calvo-Rizo Applied Sciences 2018',
        'license': 'Not established from dataset homepage; inspect distribution license before use',
        'requests': [('primusCalvoRizoAppliedSciences2018.tgz', 'https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz', 'archive_head')],
        'official_split_documentation': 'No official split specified on homepage; archive not acquired, absence inside archive unverified',
        'excluded': ['Camera-PrIMuS'],
    },
}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    for name, spec in SOURCES.items():
        folder = RAW / name
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for filename, url, kind in spec['requests']:
            target = folder / filename
            entry = dict(filename=filename, url=url, request_kind=kind,
                         attempted_at_utc=now(), downloaded_at_utc=None,
                         actual_revision=None, remote_size_bytes=None,
                         downloaded_size_bytes=0, sha256=None, status='not_downloaded')
            if target.exists():
                # Never replace an existing original; record local hash and parse.
                entry.update(downloaded_size_bytes=target.stat().st_size,
                             sha256=hashlib.file_digest(target.open('rb'), 'sha256').hexdigest(),
                             status='existing_unverified')
                if kind == 'json':
                    try:
                        json.loads(target.read_text())
                        entry['status'] = 'reused_json_readable'
                    except Exception as error:
                        entry['error'] = str(error)
            else:
                args = ['curl', '--silent', '--show-error', '--fail', '--location',
                        '--connect-timeout', '10', '--max-time', '20']
                if kind == 'archive_head':
                    args += ['--head', url]
                else:
                    args += ['--continue-at', '-', '--output', str(target) + '.partial', url]
                result = subprocess.run(args, capture_output=True, text=True)
                entry.update(curl_exit_code=result.returncode,
                             response_headers=result.stdout if kind == 'archive_head' else None,
                             error=result.stderr.strip() or None)
                if result.returncode:
                    entry['status'] = 'failed'
                elif kind == 'archive_head':
                    entry['status'] = 'metadata_only_download_pending_space_check'
                else:
                    partial = Path(str(target) + '.partial')
                    try:
                        json.loads(partial.read_text())
                        partial.rename(target)
                        entry.update(status='downloaded_json_readable', downloaded_at_utc=now(),
                                     downloaded_size_bytes=target.stat().st_size,
                                     sha256=hashlib.file_digest(target.open('rb'), 'sha256').hexdigest())
                    except Exception as error:
                        entry.update(status='failed_json_validation', error=str(error))
            partial = Path(str(target) + '.partial')
            entry['partial_size_bytes'] = partial.stat().st_size if partial.exists() else 0
            entries.append(entry)
        manifest = dict(dataset=name, absolute_path=str(folder), observed_at_utc=now(),
                        disk_free_bytes=shutil.disk_usage(RAW).free,
                        **{k: v for k, v in spec.items() if k != 'requests'}, files=entries,
                        archive_validation='not_run_no_archive',
                        image_validation='not_run_no_image', image_annotation_link_validation='not_run',
                        candidate_question_counts=None, acquired_image_count=0,
                        acquired_question_count=0, acquired_cad_model_count=0)
        (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(name, [(e['filename'], e['status'], e.get('curl_exit_code')) for e in entries], flush=True)

if __name__ == '__main__':
    main()
