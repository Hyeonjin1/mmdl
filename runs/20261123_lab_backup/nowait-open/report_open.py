"""Render the open MMMU report from frozen configuration and checkpoints."""
import json
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])
summary = json.loads((out / 'summary.json').read_text())
config = json.loads((out / 'run_config.json').read_text())
targets = {row['id']: row for row in json.loads((out / 'target_ids.json').read_text())['targets']}
catalog = json.loads((out / 'blocked_tokens.json').read_text())
blocked = {row['id']: row['decoded'] for row in catalog['blocked']}
conditions = ('baseline', 'nowait')
rows = {
    c: {r['question_id']: r for r in map(json.loads, (out / c / 'predictions.jsonl').read_text().splitlines())}
    for c in conditions
}
expected = summary['expected_pairs']
generated_pairs, judged_pairs = summary['generated_pairs'], summary['judged_pairs']
fraction = summary['token_saving_fraction']
tokens = f'{fraction:+.2%}' if fraction is not None else '미집계'

lines = [
    '# MMMU 주관식 NoWait 파일럿', '',
    f'과거 MMMU validation 900문항 중 팀 평가가 주관식으로 분류한 {expected}문항 전체를 고정했다. '
    '과거 종료 사유나 채점 결과로 필터링하지 않았다. 기존 응답은 문항 선정 및 입력 복원에 사용했다. '
    '두 조건 모두 새로 생성했다. 53문항의 원래 정답과 정규화된 허용 답안 목록을 동결했다.', '',
    '## 평가 조건', '',
    '- Baseline은 마스크 없이, NoWait는 기존 객관식의 동일한 17개 키워드와 918개 ID를 매 생성 단계에서 −∞ 로짓으로 차단했다.',
    '- Student는 Qwen3-VL-4B-Instruct다. 동일 가중치·토크나이저·팀 주관식 CoT·이미지·전처리·반복 감지를 사용했다. '
    '샘플링은 seed 3407, temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, repetition penalty 1.0이다. '
    '생성 상한 8,192토큰, 컨텍스트 49,152다. 기존 객관식 실행 설정과 대조했다.',
    '- 이미지 54개를 고정된 MMMU validation 리비전에서 RGB PNG로 복원했다. '
    '팀 원본의 문항, 정답, 이미지 이름과 순서를 대조했다. PNG와 RGB SHA-256을 고정했고 '
    '과거 원본 이미지 파일과의 바이트 동일성은 확인하지 못했다.',
    '- 주관식 채점은 기존 팀 Qwen3-8B에 허용 답안을 제공해 최종 답 추출·동치 판정을 수행했다. '
    '기존 team run.py의 open task와 score를 적용했다. 길이 종료에는 답 추출을 시도하고, '
    '반복 중단은 기존 정책대로 Judge 없이 오답 처리했다. 불확실·형식 오류는 보류했다.', '',
    '## 결과', '',
    f'생성 비교 분모는 양쪽 생성 완료 {generated_pairs}/{expected}쌍이다. '
    f'정확도 비교 분모는 양쪽 채점 완료 {judged_pairs}/{expected}쌍이다. '
    f'NoWait의 생성 토큰 절감률은 **{tokens}**다.', '',
    '| 조건 | 생성 완료 | 채점 완료 | 공통 채점 정답 | 생성 토큰 합계 | 길이 종료 | 반복 중단 | 미추출 | 채점 보류 | 차단 ID 발생 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for c in conditions:
    d = summary['conditions'][c]
    pending = sum(reason != 'not_yet_judged' for reason in d['pending'].values())
    correct = d['paired_correct']
    accuracy = f'{correct}/{judged_pairs} ({correct/judged_pairs:.1%})' if judged_pairs else '—'
    lines.append(f'| {c} | {d["generated"]}/{expected} | {d["judged"]}/{expected} | {accuracy} | '
                 f'{d["paired_generation_tokens"]:,} | {d["length"]} | {d["repetition_abort"]} | '
                 f'{d["unextracted"]} | {pending} | {d["blocked_token_matches"]:,} |')

lines += ['', '미추출은 확인된 미추출과 반복 중단을 포함한다. 보류는 미추출과 겹칠 수 있고, '
          'Judge 형식 오류로 추출 여부 자체를 확인하지 못한 경우는 미추출 수에 넣지 않았다.', '',
          '### 정답 변화 및 사례', '']


def field(value):
    return '—' if value is None else str(value).replace('\n', ' ').replace('|', '\\|').strip()[:150]


for key, label in [('gained_ids', 'NoWait에서 새로 맞힘'), ('lost_ids', 'NoWait에서 정답을 잃음')]:
    ids = summary[key]
    lines.append(f'- **{label} {len(ids)}개**. {", ".join(ids) or "없음"}')
    for q in ids[:3]:
        a, b = (rows[c][q] for c in conditions)
        lines.append(f'  - {q} · 기준 답 {field(targets[q]["gold"])} · '
                     f'Baseline {field(a["prediction"])} ({a["finish_reason"]}) → '
                     f'NoWait {field(b["prediction"])} ({b["finish_reason"]})')
        for c, row in zip(conditions, (a, b)):
            raw = row.get('judge', {}).get('raw')
            if raw:
                lines.append(f'    - {c} 판정 근거 · {field(json.loads(raw)["reason"])}')
        prediction = a['prediction']
        if key == 'lost_ids' and prediction:
            possible = sorted({blocked[i].strip() for i in a['generated_token_ids'][-128:] if i in blocked
                               and blocked[i].strip().casefold() in prediction.casefold()})
            if possible:
                lines.append(f'    - Baseline 출력 끝 128토큰에서 추출 답에 포함되는 차단 후보가 발견됐다 '
                             f'({field(", ".join(possible))}). NoWait에서 답이 바뀐 원인인지는 확인하지 못했다.')

retained = []
for q in sorted(rows['baseline'].keys() & rows['nowait'].keys()):
    a, b = (rows[c][q].get('repetition_evidence') for c in conditions)
    if a and b:
        x, y = (e['normalized_phrase'] for e in (a,b))
        words = [set(re.findall(r'\w+', text)) for text in (x,y)]
        similarity = len(words[0]&words[1])/max(1,len(words[0]|words[1]))
        retained.append((similarity,q,a,b))
for similarity, q, a, b in sorted(retained, reverse=True)[:2]:
    lines.append(f'- 반복 중단 비교 · {q} · Baseline {a["occurrences"]}회, NoWait {b["occurrences"]}회. '
                 f'감지 구절 단어 집합의 겹침 {similarity:.0%}. '
                 f'Baseline 감지 구절 앞부분 “{field(" ".join(a["normalized_phrase"].split()[:14]))}”, '
                 f'NoWait “{field(" ".join(b["normalized_phrase"].split()[:14]))}”. '
                 '구절의 유사도만으로 풀이가 같거나 같은 원인으로 반복됐다고 판정할 수 없다.')

lines += ['', '### 채점 보류', '']
for c in conditions:
    pending = summary['conditions'][c]['pending']
    lines.append(f'- {c} · {len(pending)}건' + (' · ' + ', '.join(f'{q} ({field(e)})' for q,e in pending.items()) if pending else ''))

lines += ['', '## 한계와 재현', '',
          '단일 seed와 한 Student 모델에서 얻은 결과다. 정답 추출·동치 판단은 로컬 Judge의 규약을 따른다. '
          '생성 토큰 감소율은 출력 길이를 나타내며 입력 처리, 지연 시간, GPU 비용을 따로 계측하지 않았다. '
          '문항별 원문·토큰·채점 기록은 실행 결과 디렉터리의 조건별 predictions.jsonl에 보존했다.', '',
          f'- 팀 커밋 `{config["team_commit"]}`',
          f'- 고정 마스크 SHA-256 `{config["blocked_ids_sha256"]}`',
          f'- 기존 객관식 설정 SHA-256 `{config["original_mc_config_sha256"]}`',
          '- Docker 실행기에서 smoke → generate → judge 순서로 실행하며 저장된 조건·문항을 건너뛴다.',
          '', '[실행 설정](run_config.json) · [집계 JSON](summary.json) · [고정 문항 목록](target_ids.json)', '']
(out / 'experiment.md').write_text('\n'.join(lines), encoding='utf-8')
