# Qwen3-VL-4B-Instruct MMMU NoWait 파일럿

과거 팀 B0에서 8,192토큰에 도달했고 보완 규칙으로도 답을 추출하지 못한 객관식 40문항에 한정한다. 동일한 선정 ID에서 새로 생성한 baseline과 NoWait를 비교한다.

## 재현 설정

- 논문 본문은 15개를 설명한다. 이 파일럿은 키워드 상자에 열거된 17개를 사용했다. 고정 Student 토크나이저에서 개별 일반 토큰을 디코딩하고 대소문자를 무시한 부분 문자열로 비교했다. vLLM 0.11.2 V1의 요청별 사용자 정의 프로세서가 NoWait의 매 샘플링 직전에 차단 ID를 −∞로 바꾼다. Instruct의 생성 출력 전체에 적용했다. 프롬프트와 Judge에는 마스크가 비활성화된다. [EMNLP Findings 원문](https://aclanthology.org/2025.findings-emnlp.394.pdf)
- 어휘 151,669개에서 918개 차단. 특수 ID 16개 제외. 차단 ID SHA-256 `815552d5f96d397079edc4d701858d4c36ea914fdb0eb5ae8c484466a5c51836`. 실제 부분 문자열 부수 차단 사례는 `know`(ID 1414), `Button`(ID 1567), `button`(ID 2159), `company`(ID 2813), `button`(ID 3137), `anyone`(ID 5489), `otherwise`(ID 5937), `knowledge`(ID 6540), `mother`(ID 6554), `Button`(ID 6601), `Company`(ID 8188), `company`(ID 10139).
- 팀 checkout `1c4d7b32ab254d8444b401b06750dfb6e2bfca9e`. Qwen Student `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Judge Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`. 엔진·입력·체크포인트 해시의 세부값은 `run_config.json`과 `target_ids.json`에 있다.
- 팀 B0 SHA-256 `a2336c6a92acbb190ae6c1a40085c97db942158ea107ae1db5469451c3734e39`, 과거 판정 SHA-256 `f1ff9fb18fb2949f1e64dd53df5d2a8b37d38e21258aaf2b9a70d2bb8dff0fba`, 이미지 매니페스트 SHA-256 `df0c951217114190b5cacef28a81e70746d02f8ba9549f956a4baf71ef51a6e2`. 캐시 이미지의 순서, PNG·RGB 해시를 검증했다. 과거 실행의 이미지 파일과 바이트 동일성은 미검증이다.
- 이전 Lookback 대조군 재사용은 0건이다. 컨텍스트 차이는 16,384 대 49,152이다. 이전 기록 SHA-256 `f8792124d731858e21cd59a2599e916147110389ca0a1f3e97f55322fc086928`.
- 두 Student 조건은 팀 CoT 메시지, Qwen 이미지 전처리 1,003,520/4,014,080 픽셀, 반복 감지, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0, presence penalty 1.5, 요청 seed 3407, 출력 8,192, 컨텍스트 49,152를 공유한다. Student KV 접두사 캐시는 껐고 CUDA graphs를 사용한다. 생성 시간은 엔진 호출 시작부터 종료까지다.
- 팀 Qwen3-8B Judge를 temperature 0, 최대 512토큰, context 32,768, thinking 비활성으로 사용했다. Judge에는 질문·보기·응답만 전달한다. 유효한 선택지 판정을 우선하고 불확실 판정에는 팀 terminal_rules_v1을 적용한다. 규칙 기권 및 repetition_abort는 B0 집계 정책을 따른다. 유효하지 않은 Judge 출력이 규칙으로도 해소되지 않으면 미채점 상태로 남긴다.
- CPU 합성 검증과 실제 GPU smoke의 결과는 `cpu_validation.json`, `smoke_report.json`에서 확인할 수 있다. GPU smoke 통과. 실행 직전 장비와 다른 GPU 작업을 `gpu_preflight.jsonl`에 기록한다.
- 실행 명령은 `bash code/MMDL-official-vllm/nowait.sh run results/mmmu_nowait_pilot_20261123`. 실제 단계별 Python 명령은 `commands.log`에 기록한다.

## 결과

채점 완료 쌍 2/40. 미완료 ID validation_Architecture_and_Engineering_13, validation_Architecture_and_Engineering_17, validation_Architecture_and_Engineering_18, validation_Architecture_and_Engineering_22, validation_Architecture_and_Engineering_26, validation_Architecture_and_Engineering_9, validation_Computer_Science_12, validation_Electronics_14, validation_Electronics_17, validation_Electronics_28, validation_Electronics_7, validation_Electronics_9, validation_Energy_and_Power_1, validation_Energy_and_Power_12, validation_Energy_and_Power_13, validation_Energy_and_Power_14, validation_Energy_and_Power_17, validation_Energy_and_Power_22, validation_Energy_and_Power_4, validation_Energy_and_Power_8, validation_Energy_and_Power_9, validation_Marketing_23, validation_Materials_20, validation_Materials_22, validation_Materials_4, validation_Materials_5, validation_Materials_6, validation_Materials_9, validation_Math_16, validation_Mechanical_Engineering_27, validation_Mechanical_Engineering_28, validation_Mechanical_Engineering_4, validation_Mechanical_Engineering_8, validation_Music_14, validation_Music_23, validation_Music_25, validation_Music_27, validation_Physics_18.

| 조건 | 생성/채점 | 정답 | 정확도 | 추출 | 길이 종료 | 반복 종료 | 생성 토큰 합계/평균/중앙값 | 총 생성 초 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5/2 | 0 | 0.00% | 1 | 1 | 0 | 10,279 / 5,139.50 / 5,139.5 | 118.95 |
| nowait | 4/2 | 0 | 0.00% | 1 | 0 | 1 | 7,344 / 3,672.00 / 3,672.0 | 81.54 |

순증 정답 +0개. 표와 변화량의 분모는 채점이 완료된 짝이다.

- 미추출→정답 0건. 
- 오답→정답 0건. 
- 정답→오답 0건. 
- 정답→미추출 0건. 

새 baseline 실제 길이 종료 4개 중 채점 완료 1쌍. 해당 NoWait에서 정상 종료 0개, 추출 0개, 정답 0개, 반복 종료 1개.

### 토큰 절감

| 완료 쌍 기준 | 쌍 | Baseline 토큰 | NoWait 토큰 | 절감률 |
|---|---:|---:|---:|---:|
| 전체 | 2 | 10,279 | 7,344 | 28.55% |
| 양쪽 정상 종료 | 1 | 2,087 | 1,712 | 17.97% |
| NoWait 반복 종료 | 1 | 8,192 | 5,632 | 31.25% |
| NoWait 반복 종료 제외 | 1 | 2,087 | 1,712 | 17.97% |

원본 ID 검증에서 NoWait 4문항의 위반 0건. Baseline은 원본 ID가 있는 5문항 중 5문항에서 차단 대상 토큰 총 530회가 나왔다. 측정 불가 0문항.

### 변화 사례

- 미추출→정답 0건.
- 오답→정답 0건.
- 정답→오답 0건.
- 정답→미추출 0건.

해석 범위는 과거 길이 제한 미추출 40문항이다. 부분 문자열 차단은 일반 어휘와 최종 답 표현에도 영향을 줄 수 있다.
