# Qwen3-VL-4B-Instruct MMMU Lookback 파일럿

팀 B0에서 길이 제한으로 미추출된 40문항과 정상 종료 40문항을 고정했다. 정상 종료군은 과거 팀 채점의 정답 20개와 오답 20개로 구성된다. 새로 생성한 대조군과 Lookback 단독 조건을 문항 ID로 짝지어 비교한다. 과거 잘림 사례를 의도적으로 포함한 파일럿이며 결과의 범위는 선택된 80문항이다.

## 출처와 고정 설정

- 팀 checkout `eses0070/MMDL`의 `1c4d7b32ab254d8444b401b06750dfb6e2bfca9e`. 팀 B0 `results/validation900_cot8192_guard_full/predictions.jsonl` SHA-256 `a2336c6a92acbb190ae6c1a40085c97db942158ea107ae1db5469451c3734e39`. 과거 정상 종료 판정 `results/validation900_rules_v1/judgments.jsonl` SHA-256 `f1ff9fb18fb2949f1e64dd53df5d2a8b37d38e21258aaf2b9a70d2bb8dff0fba`.
- `seokmin`의 `b5bd7c706208de056174719746511ef27588ff1f`에서 `experiments/parser_v2/results/verified_parser_v2/remaining.jsonl`을 읽었다. SHA-256 `e4a2e91a2d11e9a1d987514b00c2653bf1549ad85b3440dad46faf66970c88ed`. 진단 40개 모두 팀 B0의 `length` 행과 정답이 일치했다. 이미지가 있고 정상 종료한 과거 정답 555개, 오답 232개에서 ID를 정렬하고 하나의 `Random(3407)`으로 순서대로 20개씩 추출했다.
- 이미지 출처 `MMMU/MMMU` validation `876ce5cb130f7f7e290ce4d9984357737d4db5cf`. 로컬 캐시 이미지 매니페스트의 SHA-256은 `df0c951217114190b5cacef28a81e70746d02f8ba9549f956a4baf71ef51a6e2`. 모든 선택 ID에서 매니페스트와 팀 B0의 이미지 이름, 순서, 문항 정보 매핑을 대조했고 PNG 파일 및 RGB 픽셀 SHA-256을 검증했다. 이전 실행의 원격 경로에 있는 원본 파일과의 바이트 동일성은 미검증이다. 문항별 경로와 다이제스트는 `target_ids.json`에 있다.
- 모델 `Qwen/Qwen3-VL-4B-Instruct` `ebb281ec70b05090aa6165b016eac8ec08e71b17`. 가중치와 토크나이저 등 캐시 파일 SHA-256은 `pilot_config.json`에 고정했다. NVIDIA GeForce RTX 4090 24,564 MiB, 드라이버 595.99.02. vLLM 0.11.2, Transformers 4.57.1, BF16, 문항당 동시 요청 1, 입력과 출력 합산 컨텍스트 16,384.
- 팀 B0가 저장한 CoT 사용자 메시지에 캐시 이미지 경로만 대입했다. 두 조건 모두 기존 Qwen 전처리 `96588727e44c78b25ba03ea03b8e12f7e64fd0da`, min/max pixels 1,003,520/4,014,080, temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, repetition penalty 1.0, 문항별 생성 seed 3407, 생성·삽입 합계 8,192토큰을 사용한다.
- 채점은 추론 전에 파일과 설정 해시를 잠갔다. 선택형은 `seokmin` parser v2의 결정론적 마지막 답 라인 규칙을 사용하며 LLM 제안은 전달하지 않는다. 주관식 2문항에는 공식 MMMU `b26bdefc7e5e007962fde8f93b2bb53c99c8d13d`의 open 파서와 비교 규칙을 마지막 비어 있지 않은 줄에 적용한다. 길이/반복 종료와 파서가 기권한 응답은 미추출·오답으로 집계한다. 이 규칙은 두 신규 조건에 적용된다. 과거 채점은 문항 선정에 사용했다.

## Lookback 구현

- Algorithm 1과 Table 2에 따라 최근 모델 생성 50토큰에서 공개된 발동 문구 10개를 찾는다. 공개된 삽입 문구 10개를 별도 RNG `Random(3408)`로 문항마다 초기화해 균등 선택한다. 부록의 전체 문구 집합 크기는 각각 55개와 23개다. 실험 설정에는 공개 문구만 사용한 부분 재현을 고정했다. [CVPR 2026 공식 부록](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Bi_When_to_Think_CVPR_2026_supplemental.pdf)
- 원문 철자와 어순을 유지한 문구를 대소문자만 무시하고 부분 문자열로 매칭한다. 최초 발동은 즉시 가능하고, 재발동에는 모델 생성 토큰 기준 50 초과 간격을 요구한다. 최대 5회이며 Algorithm 1의 세 답 단계 신호 중 최초를 검출하면 이후 삽입을 차단한다. 문구는 누적 출력에서 추가된 토큰만 순서대로 처리한다. 문구 경계에 줄바꿈과 공백을 추가한 실제 삽입 토큰까지 상한에 포함한다. 진단군 과거 텍스트에서 이 규칙으로 문구 27/40건, 답 신호 이전 13/40건이 검출됐다. 요청에 제시된 점검 수치는 29/40건, 17/40건이다. 이 비교는 텍스트 문자 위치를 사용하며 실행 중 발동은 실제 토큰 위치를 사용한다.
- 기존 `streaming_guard.py`에서 누적 토큰의 첫 발동 위치까지 확정하고 진행 중 요청을 취소한다. 동일 이미지와 전처리 설정을 넣고, 재토큰화 없는 응답 토큰 ID 접두사와 삽입 ID를 붙여 재요청한다. 재요청의 엔진 확장 프롬프트 ID를 첫 요청 ID와 전체 응답 접두사의 연결로 검증한다. 이미 도착한 청크에 발동 이후 토큰이 있다면 버리고 그 수를 기록한다. 두 조건의 반복 감지에는 기존 정책을 동일하게 적용한다.
- 재요청은 KV 접두사 캐시를 이용할 수 있으며 vLLM의 생성 RNG 및 출력 토큰에 적용하는 presence penalty 이력이 새 요청에서 초기화된다. 초기 KV 캐시는 문항·조건별 salt로 격리했다. 최초 요청 캐시 적중이 0인지 양쪽에서 검증한다. 재요청별 prompt 토큰, 적중 토큰, 최초 출력까지의 wall time을 기록한다. 최초 출력 시간은 전처리 이후 요청 처리와 첫 디코딩까지 포함하는 prefill 비용의 대리지표다. 개별 순수 GPU 커널 시간은 계측하지 않았다. 문항별 전체 시간은 이미지 전처리부터 생성 완료까지이며 모델 로딩과 채점·디스크 기록은 제외한다.

## 명령과 체크포인트

작업 디렉터리 `/home/gpu/projects/MMDL-team`에서 실행한다. 정확한 Python 호출은 `commands.log`에 남는다.

```bash
bash results/mmmu_lookback_pilot_20261123/replay.sh smoke
bash results/mmmu_lookback_pilot_20261123/replay.sh run
```

두 번째 명령이 재개 명령이다. 이미 저장된 조건·문항 ID는 건너뛰고 중간에 끊긴 마지막 JSONL 줄은 다시 생성한다. 두 조건의 예측에서 개입 로그, 짝지은 결과, 집계 JSON을 재생성한다.

첫 smoke 시도는 시스템 C 컴파일러 부재로 엔진 초기화에서 실패했다. 실습실 기존 Conda GCC와 Python 헤더를 사용해 해결했다. 두 번째 시도에서 같은 문항의 직전 조건 KV가 재사용돼 개입이 없었는데도 진단 문항 생성이 달라졌다. 해당 로그와 예측은 `smoke-cross-cache.log`와 `smoke-cross-cache/`에 보존했다. 조건별 캐시 격리를 적용한 smoke와 파일럿은 새 프로토콜 해시를 사용한다.

## 완료 검증

- 실습실 본 실행의 마지막 응답은 2026-11-23 04:32 KST에 저장됐다. NCCL 종료 정리 경고가 표시된 뒤 셸로 돌아왔다. 두 조건 각 80개, 짝 80개가 고정 ID와 일치하고 개입 40개가 기록됐다. 모두 8,192토큰 상한을 지켰다. 개입 없는 61개 문항은 두 조건의 응답 텍스트와 종료 사유가 일치했다. 모든 재요청의 멀티모달 접두사 검증, 토큰 집계, 50토큰 초과 재발동 간격, 답 단계 이후 개입 차단도 확인했다.
- 정상 종료군의 정답→미추출 두 사례에는 Lookback 원문에 정답 선택 문자가 나온다. `validation_Mechanical_Engineering_19`의 답은 여러 줄 수식의 `\boxed{B...}`에 있고 마지막 줄은 `$$`였다. `validation_Math_21`은 `C` 뒤에 보기 문구를 바꿔 썼고, 파서가 문자와 보기 내용의 일치를 확정하지 못했다. 아래 집계에는 사전 고정 채점 결과를 유지했다.

<!-- generated results -->

## 실행 결과

### Smoke test

통과.

저장된 짝 2/2.

| 문항군 | 조건 | 완료 | 정확도 | 추출 | 길이 종료 | 반복 종료 | 평균 생성 / 삽입 토큰 | 평균 / 합계 초 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_length | baseline | 1/1 | 0/1 (0.0%) | 0/1 (0.0%) | 0 | 1 | 7936.0 / 0.0 | 93.36 / 93.36 |
| diagnostic_length | lookback | 1/1 | 0/1 (0.0%) | 0/1 (0.0%) | 1 | 0 | 8170.0 / 22.0 | 93.43 / 93.43 |
| normal | baseline | 1/1 | 1/1 (100.0%) | 1/1 (100.0%) | 0 | 0 | 128.0 / 0.0 | 1.45 / 1.45 |
| normal | lookback | 1/1 | 1/1 (100.0%) | 1/1 (100.0%) | 0 | 0 | 128.0 / 0.0 | 1.49 / 1.49 |

- diagnostic_length 1쌍에서 오답→정답 0건, 미추출→정답 0건, 정답→오답 0건, 정답→미추출 0건.
- diagnostic_length Lookback 1문항 개입, 총 3회 삽입. 첫 답 단계 신호 0문항, 신호 이후 매칭 0문항. 재요청 프롬프트 20326토큰, 캐시 적중으로 보고된 16144토큰, 재요청 첫 출력까지 합계 0.399초.
- normal 1쌍에서 오답→정답 0건, 미추출→정답 0건, 정답→오답 0건, 정답→미추출 0건.
- normal Lookback 0문항 개입, 총 0회 삽입. 첫 답 단계 신호 1문항, 신호 이후 매칭 0문항. 재요청 프롬프트 0토큰, 캐시 적중으로 보고된 0토큰, 재요청 첫 출력까지 합계 0.000초.

- `validation_Accounting_22` 발동 위치 `[2956, 6698, 7080]`, 첫 답 단계 신호 `미검출`.
- `validation_Art_25` 발동 위치 `[]`, 첫 답 단계 신호 `Therefore, @ 115`.

### 80문항 파일럿

저장된 짝 80/80.

| 문항군 | 조건 | 완료 | 정확도 | 추출 | 길이 종료 | 반복 종료 | 평균 생성 / 삽입 토큰 | 평균 / 합계 초 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diagnostic_length | baseline | 40/40 | 4/40 (10.0%) | 6/40 (15.0%) | 24 | 10 | 6440.7 / 0.0 | 71.69 / 2867.55 |
| diagnostic_length | lookback | 40/40 | 6/40 (15.0%) | 8/40 (20.0%) | 26 | 6 | 6415.6 / 5.5 | 71.37 / 2854.65 |
| normal | baseline | 40/40 | 21/40 (52.5%) | 39/40 (97.5%) | 0 | 0 | 1248.3 / 0.0 | 13.55 / 542.01 |
| normal | lookback | 40/40 | 19/40 (47.5%) | 37/40 (92.5%) | 1 | 0 | 1197.1 / 1.3 | 13.06 / 522.49 |

- diagnostic_length 40쌍에서 오답→정답 0건, 미추출→정답 2건, 정답→오답 0건, 정답→미추출 0건.
- diagnostic_length Lookback 13문항 개입, 총 32회 삽입. 첫 답 단계 신호 29문항, 신호 이후 매칭 15문항. 재요청 프롬프트 149242토큰, 캐시 적중으로 보고된 105360토큰, 재요청 첫 출력까지 합계 3.963초.
- normal 40쌍에서 오답→정답 0건, 미추출→정답 0건, 정답→오답 0건, 정답→미추출 2건.
- normal Lookback 6문항 개입, 총 8회 삽입. 첫 답 단계 신호 35문항, 신호 이후 매칭 2문항. 재요청 프롬프트 18361토큰, 캐시 적중으로 보고된 4704토큰, 재요청 첫 출력까지 합계 1.227초.

### 답이 바뀐 사례와 조기 차단

- `validation_Electronics_28` (unextracted->correct, 정답 `A`). 대조군 `None` / repetition_abort, Lookback `A` / stop, 개입 1회.
- `validation_Marketing_23` (unextracted->correct, 정답 `C`). 대조군 `None` / length, Lookback `C` / stop, 개입 1회.
- `validation_Mechanical_Engineering_19` (correct->unextracted, 정답 `B`). 대조군 `B` / stop, Lookback `None` / stop, 개입 1회.
- `validation_Math_21` (correct->unextracted, 정답 `C`). 대조군 `C` / stop, Lookback `None` / stop, 개입 1회.
- `validation_Architecture_and_Engineering_18`의 첫 답 신호는 생성 토큰 1779, 답 단계 이후 불확실성 문구는 2178. 개입 0회, 종료 `length`.
- `validation_Computer_Science_12`의 첫 답 신호는 생성 토큰 865, 답 단계 이후 불확실성 문구는 6706. 개입 0회, 종료 `length`.
- `validation_Energy_and_Power_14`의 첫 답 신호는 생성 토큰 1507, 답 단계 이후 불확실성 문구는 4751. 개입 0회, 종료 `length`.
