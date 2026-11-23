# MMMU 주관식 NoWait 파일럿

과거 MMMU validation 900문항 중 팀 평가가 주관식으로 분류한 53문항 전체를 고정했다. 과거 종료 사유나 채점 결과로 필터링하지 않았다. 기존 응답은 문항 선정 및 입력 복원에 사용했다. 두 조건 모두 새로 생성했다. 53문항의 원래 정답과 정규화된 허용 답안 목록을 동결했다.

## 평가 조건

- Baseline은 마스크 없이, NoWait는 기존 객관식의 동일한 17개 키워드와 918개 ID를 매 생성 단계에서 −∞ 로짓으로 차단했다.
- Student는 Qwen3-VL-4B-Instruct다. 동일 가중치·토크나이저·팀 주관식 CoT·이미지·전처리·반복 감지를 사용했다. 샘플링은 seed 3407, temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, repetition penalty 1.0이다. 생성 상한 8,192토큰, 컨텍스트 49,152다. 기존 객관식 실행 설정과 대조했다.
- 이미지 54개를 고정된 MMMU validation 리비전에서 RGB PNG로 복원했다. 팀 원본의 문항, 정답, 이미지 이름과 순서를 대조했다. PNG와 RGB SHA-256을 고정했고 과거 원본 이미지 파일과의 바이트 동일성은 확인하지 못했다.
- 주관식 채점은 기존 팀 Qwen3-8B에 허용 답안을 제공해 최종 답 추출·동치 판정을 수행했다. 기존 team run.py의 open task와 score를 적용했다. 길이 종료에는 답 추출을 시도하고, 반복 중단은 기존 정책대로 Judge 없이 오답 처리했다. 불확실·형식 오류는 보류했다.

## 결과

생성 비교 분모는 양쪽 생성 완료 53/53쌍이다. 정확도 비교 분모는 양쪽 채점 완료 50/53쌍이다. NoWait의 생성 토큰 절감률은 **21.50%**다.

| 조건 | 생성 완료 | 채점 완료 | 공통 채점 정답 | 생성 토큰 합계 | 길이 종료 | 반복 중단 | 미추출 | 채점 보류 | 차단 ID 발생 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 53/53 | 52/53 | 21/50 (42.0%) | 100,354 | 4 | 4 | 5 | 1 | 1,441 |
| nowait | 53/53 | 51/53 | 20/50 (40.0%) | 78,775 | 1 | 1 | 1 | 2 | 0 |

미추출은 확인된 미추출과 반복 중단을 포함한다. 보류는 미추출과 겹칠 수 있고, Judge 형식 오류로 추출 여부 자체를 확인하지 못한 경우는 미추출 수에 넣지 않았다.

### 정답 변화 및 사례

- **NoWait에서 새로 맞힘 0개**. 없음
- **NoWait에서 정답을 잃음 1개**. validation_Manage_17
  - validation_Manage_17 · 기준 답 65 · Baseline 65 (stop) → NoWait $85.76 (stop)
    - baseline 판정 근거 · The candidate's final answer matches the accepted answer of 65.
    - nowait 판정 근거 · The accepted answer is $65, but the candidate's final answer is $85.76, which does not match.

Baseline은 수수료 \$85.75625를 계산한 뒤 \$65 상한을 적용했다. NoWait는 같은 중간값을 반올림한 \$85.76을 최종 답으로 써 상한을 적용하지 않았다. 특정 차단 토큰이 이 차이를 만들었다는 근거는 없다. Baseline은 820토큰, NoWait는 1,346토큰을 사용했다.

`validation_Electronics_29`는 NoWait에서 차단 ID가 0개였지만, 노드 A로 들어오고 나가는 전류를 설명하는 구절이 7회 반복돼 반복 감지로 중단됐다. Baseline은 같은 문항에서 Judge가 불확실 판정을 내렸다. 이 문항은 공통 정확도 분모에서 빠졌다.

### 채점 보류

- baseline · 1건 · validation_Electronics_29 (open_uncertain)
- nowait · 2건 · validation_Architecture_and_Engineering_14 (JSONDecodeError: Expecting value: line 1 column 1 (char 0)), validation_Chemistry_23 (open_uncertain)

## 한계와 재현

단일 seed와 한 Student 모델에서 얻은 결과다. 정답 추출·동치 판단은 로컬 Judge의 규약을 따른다. 생성 토큰 감소율은 출력 길이를 나타내며 입력 처리, 지연 시간, GPU 비용을 따로 계측하지 않았다. 문항별 원문·토큰·채점 기록은 실행 결과 디렉터리의 조건별 predictions.jsonl에 보존했다.

- 팀 커밋 `1c4d7b32ab254d8444b401b06750dfb6e2bfca9e`
- 고정 마스크 SHA-256 `815552d5f96d397079edc4d701858d4c36ea914fdb0eb5ae8c484466a5c51836`
- 기존 객관식 설정 SHA-256 `b2afd1fc11ec07a31d2fffe878cb50e6a46452fd07bc14a43816b4bc28d12fec`
- Docker 실행기에서 smoke → generate → judge 순서로 실행하며 저장된 조건·문항을 건너뛴다.

[실행 설정](run_config.json) · [집계 JSON](summary.json) · [문항 ID 목록](question_ids.json) · [실행 코드](nowait_open.py) · [이미지 복원 코드](restore_open_images.py)
