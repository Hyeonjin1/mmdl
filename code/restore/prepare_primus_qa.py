"""Deterministic PrIMuS QA derivation. CPU only; never starts training.

Source annotations stay in audit fields, never in the model question. These are
custom QA and custom splits, not official PrIMuS QA or an official partition.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SEED = 3407
NOTE = re.compile(r'note-([A-G])([#b]*)(-?\d+)_(.+)')
NATURAL = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)
DURATION_BASES = ['quadruple_whole', 'double_whole', 'whole', 'half', 'quarter',
                  'eighth', 'sixteenth', 'thirty_second', 'sixty_fourth', 'hundred_twenty_eighth']
CLEFS = ['G1', 'G2', 'F3', 'F4', 'F5', 'C1', 'C2', 'C3', 'C4', 'C5']
PROMPTS = {
    'first_note_pitch': 'What is the written pitch of the first non-grace note? Include the octave and apply the key signature and accidentals. Use scientific pitch notation with middle C = C4.',
    'first_note_duration': 'What is the written duration of the first non-grace note? Include augmentation dots. Ignore extra holding from fermatas, ties, and tuplet timing.',
    'initial_clef': 'Which clef appears at the beginning of the staff? Staff lines are numbered from bottom (1) to top (5).',
    'first_two_note_interval': 'What is the signed pitch change in semitones from the first non-grace note to the second non-grace note? Apply the key signature and accidentals. Positive means ascending; negative means descending.',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def duration_label(code):
    code = code.removesuffix('_fermata')
    base = code.rstrip('.')
    if base not in DURATION_BASES or code[len(base):] not in ['', '.', '..']:
        raise ValueError('Unsupported duration ' + code)
    dots = len(code) - len(base)
    return ('double-dotted ' if dots == 2 else 'dotted ' if dots == 1 else '') + base.replace('_', '-') + ' note'


def note_value(token):
    m = NOTE.fullmatch(token)
    if not m:
        raise ValueError('Unsupported note ' + token)
    letter, accidental, octave, duration = m.groups()
    octave = int(octave)
    midi = 12 * (octave + 1) + NATURAL[letter] + accidental.count('#') - accidental.count('b')
    return {'pitch': f'{letter}{accidental}{octave}', 'midi': midi,
            'letter': letter, 'octave': octave, 'duration': duration_label(duration)}


def pitch_label(midi):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return names[midi % 12] + str(midi // 12 - 1)


def options_for(task, correct, record):
    rng = random.Random(int(sha(f'{SEED}:{record["original_key"]}:{task}'.encode()), 16))
    if task == 'first_note_pitch':
        candidates = [pitch_label(record['notes'][0]['midi'] + delta) for delta in [-12, -2, -1, 1, 2, 12]]
    elif task == 'first_note_duration':
        candidates = [duration_label(base + dots) for base in DURATION_BASES for dots in ['', '.', '..']]
    elif task == 'initial_clef':
        candidates = [f'{c[0]} clef on line {c[1]}' for c in CLEFS]
    else:
        value = record['notes'][1]['midi'] - record['notes'][0]['midi']
        candidates = [str(value + delta) + ' semitones' for delta in [-12, -2, -1, 1, 2, 12]]
    candidates = sorted(set(candidates) - {correct})
    opts = [correct] + rng.sample(candidates, 3)
    rng.shuffle(opts)
    assert len(set(opts)) == 4 and opts.count(correct) == 1
    return opts, chr(65 + opts.index(correct))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets/raw/PrIMuS/extracted')
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/processed/PrIMuS_QA_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; verify/reuse it or choose a new version. No overwrite.')
    args.output.mkdir(parents=True)
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        trail = []
        while parent[key] != key:
            trail.append(key)
            key = parent[key]
        for item in trail:
            parent[item] = key
        return key

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    records = []
    rejected = []
    seen_ids = set()
    paths = sorted(p for p in args.source.rglob('*.semantic')
                   if not p.name.startswith('._') and '__MACOSX' not in p.parts)
    for index, path in enumerate(paths):
        try:
            key = path.stem
            if key in seen_ids:
                raise ValueError('Duplicate source identifier')
            seen_ids.add(key)
            tokens = path.read_text().split()
            notes = [note_value(t) for t in tokens if t.startswith('note-')][:2]
            if not notes:
                raise ValueError('No non-grace note')
            clef = next(t.removeprefix('clef-') for t in tokens if t.startswith('clef-'))
            if clef not in CLEFS:
                raise ValueError('Unsupported initial clef')
            linked = {ext: path.with_suffix(ext) for ext in ['.png', '.agnostic', '.mei']}
            linked['.pae'] = path.parent / 'regular_pae.pae'
            if not all(p.is_file() for p in linked.values()):
                raise ValueError('Missing linked original')
            agnostic = linked['.agnostic'].read_text().split()
            if not agnostic:
                raise ValueError('Empty agnostic annotation')
            with Image.open(linked['.png']) as image:
                size = list(image.size)
                image.verify()
            work = key.split('-', 1)[0]
            semantic_hash = sha(' '.join(tokens).encode())
            image_hash = file_sha(linked['.png'])
            union('work:' + work, 'semantic:' + semantic_hash)
            union('work:' + work, 'image:' + image_hash)
            records.append(dict(original_key=key, work_key=work, notes=notes, clef=clef,
                                first_note_tokens=[t for t in tokens if t.startswith('note-')][:2],
                                annotation=str(path), image=str(linked['.png']), image_size=size,
                                semantic_sha256=semantic_hash, image_sha256=image_hash))
        except Exception as error:
            rejected.append({'source': str(path), 'reason': str(error)})
        if (index + 1) % 10000 == 0:
            print('SOURCE_VALIDATED', index + 1, 'rejected', len(rejected), flush=True)

    if not records:
        raise ValueError('No valid source records; inspect source layout before claiming successful preparation')
    counts = {split: Counter() for split in ['train', 'val', 'test']}
    answers = {split: Counter() for split in counts}
    groups = defaultdict(set)
    audit_samples = random.Random(SEED).sample(records, min(256, len(records)))
    mei_checks = Counter()
    for record in audit_samples:
        tree = ET.parse(Path(record['annotation']).with_suffix('.mei'))
        elements = list(tree.iter())
        note = next(e for e in elements if e.tag.split('}')[-1] == 'note' and 'grace' not in e.attrib)
        first = record['notes'][0]
        if note.get('pname', '').upper() != first['letter'] or int(note.get('oct', '-99')) != first['octave']:
            raise ValueError('MEI first note disagrees: ' + record['original_key'])
        staff = next(e for e in elements if e.tag.split('}')[-1] == 'staffDef')
        if staff.get('clef.shape', '') + staff.get('clef.line', '') != record['clef']:
            raise ValueError('MEI clef disagrees: ' + record['original_key'])
        duration_codes = {'1': 'whole', '2': 'half', '4': 'quarter', '8': 'eighth', '16': 'sixteenth',
                          '32': 'thirty_second', '64': 'sixty_fourth', '128': 'hundred_twenty_eighth',
                          'breve': 'double_whole', 'long': 'quadruple_whole'}
        base = duration_codes.get(note.get('dur'))
        if base and duration_label(base + '.' * int(note.get('dots', '0'))) != first['duration']:
            raise ValueError('MEI duration disagrees: ' + record['original_key'])
        mei_checks['letter_octave_clef'] += 1
        mei_checks['duration'] += bool(base)
        with Image.open(record['image']) as image:
            image.load()

    image_work_split = {}
    ids = set()
    sample_rows = []
    split_handles = {s: (args.output / (s + '.jsonl.partial')).open('w') for s in counts}
    try:
        for record in records:
            group = find('work:' + record['work_key'])
            bucket = int(sha(f'{SEED}:{group}'.encode())[:16], 16) % 10000
            split = 'train' if bucket < 8000 else 'val' if bucket < 9000 else 'test'
            groups[split].add(group)
            for grouping in [record['work_key'], record['semantic_sha256'], record['image_sha256']]:
                if grouping in image_work_split and image_work_split[grouping] != split:
                    raise AssertionError('Split leakage')
                image_work_split[grouping] = split
            targets = {'first_note_pitch': record['notes'][0]['pitch'],
                       'first_note_duration': record['notes'][0]['duration'],
                       'initial_clef': f'{record["clef"][0]} clef on line {record["clef"][1]}'}
            if len(record['notes']) == 2:
                targets['first_two_note_interval'] = str(record['notes'][1]['midi'] - record['notes'][0]['midi']) + ' semitones'
            for task, correct in targets.items():
                opts, answer = options_for(task, correct, record)
                row = {k: record[k] for k in ['original_key', 'image', 'image_size', 'image_sha256', 'annotation', 'semantic_sha256']}
                row.update(id=f'PrIMuS:{record["original_key"]}:{task}', source='PrIMuS', task_type=task,
                           split=split, split_origin='custom_work_semantic_image_group_hash_seed3407', group_id=group,
                           question=PROMPTS[task] + '\nOptions:\n' + '\n'.join(f'{chr(65+i)}. {o}' for i, o in enumerate(opts)),
                           options=opts, answer=answer, source_answer=correct,
                           answer_derivation={'method': 'deterministic_semantic_parser_v1',
                                              'tokens': record['first_note_tokens'] if task != 'initial_clef' else ['clef-' + record['clef']]},
                           annotation_used_as_model_input=False)
                if row['id'] in ids:
                    raise AssertionError('Duplicate QA id')
                ids.add(row['id'])
                assert opts[ord(answer) - 65] == correct
                split_handles[split].write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[split][task] += 1
                answers[split][answer] += 1
                if len(sample_rows) < 24:
                    sample_rows.append(row)
    finally:
        for handle in split_handles.values():
            handle.close()
    files = {}
    for split in counts:
        path = args.output / (split + '.jsonl')
        path.with_suffix('.jsonl.partial').rename(path)
        files[split] = dict(path=str(path), count=sum(counts[split].values()), size_bytes=path.stat().st_size,
                            sha256=file_sha(path), tasks=dict(counts[split]), answer_letters=dict(answers[split]),
                            connected_groups=len(groups[split]))
    (args.output / 'rejected.json').write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'review_samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sample_rows))
    archive = args.source.parent / 'primusCalvoRizoAppliedSciences2018.tgz'
    manifest = dict(version='PrIMuS_QA_v1', created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    script_sha256=file_sha(Path(__file__)), seed=SEED, source_dir=str(args.source),
                    source_archive_sha256=file_sha(archive) if archive.exists() else None,
                    original_annotations=len(paths), retained_incipits=len(records), rejected=len(rejected),
                    official_qa=False, official_split=False, custom_split_target_ratio=[0.8, 0.1, 0.1],
                    grouping='Connected components of filename prefix before first hyphen, exact normalized semantic content, and exact PNG hash',
                    mei_sample_checks=dict(mei_checks), files=files, split_leakage_detected=False,
                    generated_qa_count=len(ids), training_started=False,
                    license_status='Unresolved in source homepage; derived data kept local, do not publish dataset',
                    limitations=['No chord QA; source is monophonic', 'Synthetic template QA, not human-authored QA',
                                 'First notes and initial clef only; whole-score understanding untested',
                                 'Interval uses equal-tempered chromatic semitones, not interval quality',
                                 'Written duration only; no tempo, fermata length, tied total, or tuplet timing inference',
                                 'Work groups inferred from source filenames; near-duplicate/transposed melodies not exhaustively detected',
                                 'MEI sample checks cover letter, octave, clef and duration; accidental/key-signature semantics inherited from semantic source',
                                 'Row schema matches MMDL MCQ fields; E8 training config/runner integration is not performed'])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
