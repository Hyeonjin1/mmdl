# MMMU NoWait 보고서

과거 팀 B0에서 8,192토큰 한도에 도달했고 Parser v2로도 최종 선택지를 추출하지 못한 객관식 40문항을 NoWait으로 재추론했다. **정답 8건**을 확인했고, 2건은 채점 보류였다.

## 객관식 40문항 결과

| 지표 | 과거 팀 B0 | NoWait |
| --- | ---: | ---: |
| 확인된 정답 | 0/40 | **8/40** |
| 생성 토큰 합계 | 327,680 | **245,360 (−25.12%)** |
| 8,192토큰 한도 종료 | 40 | 20 |
| 반복 중단 | 0 | 4 |

NoWait이 정답을 맞힌 8문항의 생성량은 과거 B0 65,536토큰에서 24,995토큰으로 줄었다(−61.86%). 이 행은 NoWait 정답으로 판정된 8문항만 집계한 값이다. 전체 40문항의 결과와 문항별 기록은 아래 자료에 연결했다.

[과거 B0 설정](https://github.com/eses0070/MMDL/blob/1c4d7b32ab254d8444b401b06750dfb6e2bfca9e/results/validation900_cot8192_guard_full/run_config.json) · [문항별 토큰 집계](nowait-input-audit.json) · [실행 설정](nowait-snapshot/run_config.json)

## 적용 방법

Qwen3-VL-4B-Instruct의 생성 과정에서 자기검토 표현과 철자가 겹치는 토큰의 로짓을 억제했다. MMMU validation 고정 리비전 `876ce5cb130f7f7e290ce4d9984357737d4db5cf`에서 이미지를 복원했다. 과거 B0 기록과 문항 ID, 이미지 이름·순서, 입력 토큰 개수를 대조했다. 새 응답은 Qwen3-8B Judge와 최종 답 추출 규칙으로 채점했다. 과거 B0와 NoWait의 수치는 저장 응답과 재추론 응답의 비교다.

같은 40문항에 4,096토큰 답안 전환을 적용한 후속 실험에서는 **정답 19건**을 확인했고 생성량은 과거 B0 대비 **56.16% 감소**했다. [후속 실험 보고서](nowait-followup/experiment.md)

## 주관식 53문항 결과

NoWait은 53문항 생성을 완료했다. 정답 20건을 확인했고 2건은 채점 보류였다. 생성 토큰은 78,775개였고 길이 종료와 반복 중단이 각 1건이었다. [주관식 실행 기록](nowait-open/experiment.md) · [집계 JSON](nowait-open/summary.json)
