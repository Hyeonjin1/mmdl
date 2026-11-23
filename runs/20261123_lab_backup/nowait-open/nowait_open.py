"""Run the frozen MMMU NoWait protocol on all 53 historical open questions."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path

from PIL import Image
import nowait_pilot as pilot
from checkpointing import atomic_json, contract_hash

SOURCE = Path('/work/results/mmmu_nowait_pilot_20261123')
OUT = Path('/work/results/mmmu_nowait_open_20261123')
INPUT = Path('/work/input-open')
SMOKE = ('validation_Chemistry_25', 'validation_Computer_Science_26', 'validation_Psychology_20')


def prepare():
    _, _, base, catalog = pilot.prepare(SOURCE)
    sampling = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                    presence_penalty=1.5, max_tokens=8192, seed=3407, stop_token_ids=[])
    if (base['student_sampling'] != sampling or base['student_engine']['max_model_len'] != 49152
            or base['student_engine']['seed'] != 3407 or len(base['mask']['keywords']) != 17
            or catalog['blocked_count'] != 918 or base['blocked_ids_sha256'] != catalog['blocked_ids_sha256']):
        raise ValueError('Recorded MC generation settings differ from the requested open protocol')
    if pilot.sha(pilot.OLD_JUDGE_CONFIG) != base['historical_judge_config_sha256']:
        raise ValueError('Historical Judge deployment changed')
    historical = pilot.indexed(pilot.B0)
    historical_judgments = pilot.indexed(pilot.OLD_JUDGE_CONFIG.with_name('judgments.jsonl'))
    examples = {}
    manifest_path = INPUT / 'image_manifest.json'
    manifest = pilot.read_json(manifest_path)
    if (manifest['dataset'], manifest['revision'], manifest['split'], manifest['historical_predictions_sha256']) != (
            'MMMU/MMMU', '876ce5cb130f7f7e290ce4d9984357737d4db5cf', 'validation', pilot.B0_SHA):
        raise ValueError('Open image manifest provenance mismatch')
    items = {r['id']: r for r in manifest['items']}
    open_ids = [qid for qid, row in historical.items() if row['annotation']['question_type'] == 'open']
    if (len(historical) != 900 or len(open_ids) != 53 or len(manifest['items']) != 53
            or set(items) != set(open_ids) or not set(SMOKE) <= set(open_ids)):
        raise ValueError('Expected all 53 historical open questions')
    targets = []
    for qid in open_ids:
        old, item = historical[qid], items[qid]
        annotation = old['annotation']
        task = pilot.task_for(old)
        if (task['kind'] != 'open' or item['gold'] != annotation['answer']
                or historical_judgments[qid]['gold'] != annotation['answer']):
            raise ValueError(f'Open scorer or gold mismatch for {qid}')
        messages = copy.deepcopy(old['messages'])
        parts = messages[-1]['content']
        images = [part for part in parts if part['type'] == 'image']
        if (not parts[-1]['text'].endswith(pilot.OFFICIAL_COT_PROMPT)
                or [p['image'] for p in images] != annotation['image_path']
                or [Path(p['image']).name for p in images] != [r['name'] for r in item['images']]):
            raise ValueError(f'Team prompt or image order differs for {qid}')
        for part, entry in zip(images, item['images']):
            path = INPUT / 'images' / entry['name']
            if ([part['min_pixels'], part['max_pixels']] != [1003520, 4014080]
                    or pilot.sha(path) != entry['png_sha256']):
                raise ValueError(f'Image preprocessing or bytes changed for {qid}')
            with Image.open(path) as image:
                if list(image.size) != entry['size'] or hashlib.sha256(image.convert('RGB').tobytes()).hexdigest() != entry['rgb_sha256']:
                    raise ValueError(f'Image pixels changed for {qid}')
            part['image'] = str(path)
        examples[qid] = old, messages
        targets.append(dict(id=qid, gold=item['gold'], accepted_answers=task['accepted_answers'],
                            images=item['images'], messages_sha256=contract_hash(messages)))
    pilot.freeze(OUT / 'target_ids.json', {'targets': targets, 'smoke_ids': list(SMOKE)})
    pilot.freeze(OUT / 'blocked_tokens.json', catalog)
    config = {key: base[key] for key in (
        'paper', 'team_commit', 'upstream_qwen_commit', 'student_engine', 'student_sampling',
        'student_revision', 'student_files', 'image_preprocessing', 'repetition_guard', 'environment',
        'judge_engine', 'judge_sampling', 'judge_thinking', 'judge_system', 'judge_files', 'mask', 'blocked_ids_sha256',
        'historical_judge_config_sha256', 'code_sha256')}
    config.update(original_mc_config_sha256=pilot.sha(SOURCE / 'run_config.json'),
                  historical_predictions_sha256=pilot.B0_SHA, target_ids_sha256=pilot.sha(OUT / 'target_ids.json'),
                  image_manifest_sha256=pilot.sha(manifest_path), blocked_tokens_file_sha256=pilot.sha(OUT / 'blocked_tokens.json'),
                  model_prompt='historical team open-question messages and CoT; remapped RGB PNG image paths',
                  selection='All 53 historical open rows from 900 validation questions. No finish or extraction filter.',
                  resume_code_sha256=pilot.sha(Path(__file__)),
                  scoring='Historical team run.py task_for and score using Qwen3-8B open task and no MC fallback.',
                  termination_policy={'length': 'score via historical open path', 'repetition_abort': 'wrong without Judge',
                                      'uncertain': 'pending', 'invalid_judge_output': 'pending'},
                  settings_differences_from_mc=['open-question prompt from historical rows', 'open Judge and no MC fallback',
                                                '53 open question IDs and independently exported pinned images'])
    pilot.freeze(OUT / 'run_config.json', config)
    return examples, config, catalog, contract_hash(config)


def judge_open(run, old, gen):
    if gen['finish_reason'] == 'repetition_abort':
        return dict(prediction=None, extracted=False, correct=False, answer_correct=None,
                    scoring_route='repetition_abort_policy', judge_skipped=True)
    task = pilot.task_for(dict(annotation=old['annotation'], result={'gen': gen['gen']}))
    if task['kind'] != 'open':
        raise ValueError('The open-only run cannot score a multiple-choice task')
    judged = run(task)
    prediction = None
    try:
        if judged['finish_reason'] != 'stop':
            raise ValueError('Judge output truncated')
        record = json.loads(judged['raw'])
        correct = pilot.score(record, task, old['annotation']['answer'])
        prediction = record['answer']
        if correct is None:
            return dict(scoring_route='judge_error_pending', judge_error='open_uncertain',
                        prediction=prediction, extracted=bool(prediction), judge=judged)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        return dict(scoring_route='judge_error_pending', judge_error=f'{type(error).__name__}: {error}',
                    prediction=None, extracted=None, judge=judged)
    return dict(prediction=prediction, extracted=True, correct=correct, answer_correct=correct,
                scoring_route='team_judge_open', judge_skipped=False, judge=judged)


def report(ids, generated, judgments):
    from nowait_report import jsonl
    failure_path = OUT / 'judge_failures.jsonl'
    failure_rows = [json.loads(line) for line in failure_path.read_text().splitlines()] if failure_path.exists() else []
    last_errors = {(row['condition'], row['question_id']): row for row in failure_rows}
    generated_pairs = [q for q in ids if all(q in generated[c] for c in pilot.CONDITIONS)]
    judged_pairs = [q for q in ids if all(q in judgments[c] for c in pilot.CONDITIONS)]
    summary = dict(expected_pairs=len(ids), generated_pairs=len(generated_pairs), judged_pairs=len(judged_pairs), conditions={})
    for c in pilot.CONDITIONS:
        g, j = generated[c], judgments[c]
        pending = {q: last_errors.get((c, q), {}).get('judge_error', 'not_yet_judged') for q in g if q not in j}
        summary['conditions'][c] = dict(generated=len(g), judged=len(j), correct=sum(j[q]['correct'] for q in j),
                                        paired_correct=sum(j[q]['correct'] for q in judged_pairs),
                                        length=sum(r['finish_reason']=='length' for r in g.values()),
                                        repetition_abort=sum(r['finish_reason']=='repetition_abort' for r in g.values()),
                                        unextracted=sum(not r['extracted'] for r in j.values()) +
                                                    sum(last_errors.get((c, q), {}).get('extracted') is False for q in pending),
                                        pending=pending,
                                        paired_generation_tokens=sum(g[q]['generated_tokens'] for q in generated_pairs),
                                        blocked_token_matches=sum(r['blocked_token_matches'] for r in g.values()))
        merged = []
        for q in ids:
            if q not in g:
                continue
            verdict = j.get(q)
            if verdict is None:
                failure = last_errors.get((c, q), {})
                verdict = dict(prediction=failure.get('prediction'), extracted=failure.get('extracted'),
                               correct=None, judge_error=pending[q], scoring_route=failure.get('scoring_route'))
            merged.append({**g[q], **verdict, 'scored': q in j})
        jsonl(OUT / c / 'predictions.jsonl', merged)
    a, b = (summary['conditions'][c] for c in pilot.CONDITIONS)
    base_tokens = a['paired_generation_tokens']
    summary['token_saving_fraction'] = 1-b['paired_generation_tokens']/base_tokens if base_tokens else None
    summary['gained_ids'] = [q for q in judged_pairs if not judgments['baseline'][q]['correct'] and judgments['nowait'][q]['correct']]
    summary['lost_ids'] = [q for q in judged_pairs if judgments['baseline'][q]['correct'] and not judgments['nowait'][q]['correct']]
    atomic_json(OUT / 'summary.json', summary)


def indexed(kind, ids, digest, generated=None):
    result = {}
    for c in pilot.CONDITIONS:
        (OUT / c).mkdir(exist_ok=True)
        rows = pilot.indexed(OUT / c / f'{kind}.jsonl', repair=True)
        allowed = ids if generated is None else generated[c]
        if set(rows) - set(allowed) or any(r['contract'] != digest or r['condition'] != c for r in rows.values()):
            raise ValueError(f'Invalid {kind} checkpoint for {c}')
        result[c] = rows
    return result


def grade(ids, examples, generated, judgments, config, digest, publish):
    pilot.blind_judge = judge_open
    try:
        pilot.grade(OUT, ids, examples, generated, judgments, config, digest, publish)
    except RuntimeError as error:
        if not str(error).startswith('Unscored Judge failures, can resume:'):
            raise
        print(error, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'generate', 'judge'))
    stage = parser.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.checkpoint.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        examples, config, catalog, digest = prepare()
        ids = list(examples)
        print(f'Frozen 53 open IDs, evaluation contract {digest}', flush=True)
        if stage == 'prepare':
            return
        generated = indexed('generations', ids, digest)
        judgments = indexed('judgments', ids, digest, generated)
        def publish():
            report(ids, generated, judgments)
        if stage == 'smoke':
            pilot.generate(OUT, SMOKE, examples, config, catalog, generated, digest, publish)
            grade(SMOKE, examples, generated, judgments, config, digest, publish)
            for q in SMOKE:
                assert generated['nowait'][q]['blocked_token_matches'] == 0
                for key in ('messages_sha256', 'ordered_image_sha256', 'request_sampling', 'prompt_token_sha256'):
                    assert generated['baseline'][q][key] == generated['nowait'][q][key]
            if any(not any(judgments[c].get(q, {}).get('scoring_route') == 'team_judge_open' for q in SMOKE)
                   for c in pilot.CONDITIONS):
                raise RuntimeError('Smoke did not validate open Judging in both conditions')
            pilot.freeze(OUT / 'smoke.json', dict(passed=True, ids=list(SMOKE), contract=digest))
        else:
            assert pilot.read_json(OUT / 'smoke.json')['contract'] == digest
            if stage == 'generate':
                pilot.generate(OUT, ids, examples, config, catalog, generated, digest, publish)
            else:
                if any(set(generated[c]) != set(ids) for c in pilot.CONDITIONS):
                    raise RuntimeError('Generation is incomplete')
                grade(ids, examples, generated, judgments, config, digest, publish)
        publish()


if __name__ == '__main__':
    main()
