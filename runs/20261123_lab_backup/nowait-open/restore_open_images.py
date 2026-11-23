"""Export RGB PNGs for the 53 historical open MMMU validation questions."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess

from PIL import Image
import pyarrow.parquet as pq
from export_local_mmmu import annotation

TEAM = Path('/home/gpu/projects/MMDL-team')
OUT = Path('/work/input-open')
PARQUET = Path('/work/parquet')
DATASET = 'MMMU/MMMU'
REVISION = '876ce5cb130f7f7e290ce4d9984357737d4db5cf'
B0 = TEAM / 'results/validation900_cot8192_guard_full/predictions.jsonl'
B0_SHA = 'a2336c6a92acbb190ae6c1a40085c97db942158ea107ae1db5469451c3734e39'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore(task):
    subject, historical = task
    path = PARQUET / f'{subject}.parquet'
    if not path.exists():
        temporary = path.with_suffix('.open-partial')
        url = (f'https://huggingface.co/datasets/{DATASET}/resolve/'
               f'{REVISION}/{subject}/validation-00000-of-00001.parquet?download=true')
        subprocess.run(['curl', '-fLsS', '--retry', '3', '-o', str(temporary), url], check=True)
        temporary.replace(path)
    parquet = pq.ParquetFile(path)
    columns = ['id', 'question', 'answer', 'question_type']
    columns += [name for name in parquet.schema_arrow.names if name.startswith('image_')]
    samples = {row['id']: row for row in parquet.read(columns=columns).to_pylist()}
    if len(samples) != 30:
        raise ValueError(f'Expected 30 validation questions in {subject}')
    items = []
    for old in historical:
        row, numbers = annotation(samples[old['index']], subject)
        for key in ('index', 'question', 'answer', 'question_type', 'category', 'split'):
            if row[key] != old[key]:
                raise ValueError(f'Historical {key} mismatch for {old["index"]}')
        names = [f'{row["index"]}_{number}.png' for number in numbers]
        if names != [Path(p).name for p in old['image_path']]:
            raise ValueError(f'Image order mismatch for {old["index"]}')
        images = []
        for number, name in zip(numbers, names):
            image = samples[row['index']][f'image_{number}']
            with Image.open(io.BytesIO(image['bytes'])) as source:
                rgb = source.convert('RGB')
            encoded = io.BytesIO()
            rgb.save(encoded, format='PNG')
            data = encoded.getvalue()
            target = OUT / 'images' / name
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Existing image differs: {target}')
            if not target.exists():
                temp = target.with_suffix('.partial')
                temp.write_bytes(data)
                temp.replace(target)
            images.append(dict(name=name, size=list(rgb.size), png_sha256=sha(data),
                               rgb_sha256=sha(rgb.tobytes())))
        items.append(dict(id=row['index'], gold=row['answer'], images=images))
        print(f'Verified {row["index"]}: {len(images)} images', flush=True)
    return items


def main():
    data = B0.read_bytes()
    if sha(data) != B0_SHA:
        raise ValueError('Historical baseline hash mismatch')
    rows = [json.loads(line) for line in data.splitlines()]
    selected = [r for r in rows if r['annotation']['question_type'] == 'open']
    if len(rows) != 900 or len(selected) != 53 or len({r['question_id'] for r in rows}) != 900:
        raise ValueError('Expected 53 open questions among 900 unique validation IDs')
    subjects = defaultdict(list)
    for row in selected:
        subjects[row['annotation']['category']].append(row['annotation'])
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    PARQUET.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        index = {r['id']: r for group in pool.map(restore, subjects.items()) for r in group}
    manifest = dict(dataset=DATASET, revision=REVISION, split='validation',
                    historical_predictions_sha256=B0_SHA, historical_image_bytes_verified=False,
                    items=[index[r['question_id']] for r in selected])
    path = OUT / 'image_manifest.json'
    data = json.dumps(manifest, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text() != data:
        raise ValueError('Frozen open image manifest differs')
    path.write_text(data)
    print(f'Frozen 53 open IDs, {sum(len(item["images"]) for item in manifest["items"])} images')


if __name__ == '__main__':
    main()
