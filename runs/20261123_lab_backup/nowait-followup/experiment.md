# MMMU NoWait 후속 실험

기존 NoWait 40문항을 기준으로 차단 목록 정제와 답안 전환을 각각 실행했다. Student는 동일한 Qwen3-VL-4B-Instruct이며, 첫 요청의 확장 프롬프트 토큰 SHA-256을 기존 NoWait와 대조했다.

| 조건 | 생성 토큰 합계 | 기존 대비 절감률 | 공통 채점 정답 | 채점 완료 | 채점된 응답 중 미추출 | 길이 종료 | 반복 중단 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 NoWait | 245,360 | +0.00% | 8/37 | 38/40 | 22 | 20 | 4 |
| 차단 목록 정제 | 244,736 | +0.25% | 8/37 | 38/40 | 24 | 22 | 5 |
| 4,096토큰 답안 전환 | 143,648 | +41.45% | 17/37 | 39/40 | 0 | 0 | 0 |

세 조건 모두 판정된 문항을 정확도 비교의 공통 분모로 썼다. 토큰은 생성이 끝난 40문항 전체를 합산했다.

## 구현

- 차단 목록 정제는 디코드 후 공백과 대소문자를 정규화해 사전 고정 예외 (anyone, button, company, know, knowledge, mother, ohio)에 정확히 일치하는 토큰만 제외했다. 마스크는 918개에서 886개로 줄었다. 생성 예산과 채점 조건은 기존과 동일하다.
- 답안 전환은 918개 원래 마스크에서 4,096토큰 한도까지 첫 요청을 실행한다. 그 전에 종료하면 기존 흐름을 유지하고, 한도에 도달하면 원래 이미지와 생성 토큰을 붙여 최종 선택지만 요청한다. 추가 생성 한도는 128토큰이다. 이어받은 멀티모달 프롬프트의 토큰을 대조했다. 전환 후에는 두 번째 요청의 출력으로 답을 채점했다.
- 답안 전환은 29문항에서 발동했다. 재요청 지시문은 합계 1,102 입력 토큰이며, 재요청 프롬프트에는 이미지와 기존 생성 토큰을 포함해 합계 156,034 입력 토큰이 들어갔다. 이 입력 비용은 표의 생성 토큰에서 제외한다.

## 정답 변화와 한계

- **refined_mask** 정답 획득 1개, 손실 1개. 획득 ID는 validation_Architecture_and_Engineering_17. 손실 ID는 validation_Architecture_and_Engineering_18. 채점 보류는 2건 (validation_Architecture_and_Engineering_26, validation_Materials_9).
- **answer_switch** 정답 획득 9개, 손실 0개. 획득 ID는 validation_Accounting_22, validation_Architecture_and_Engineering_9, validation_Electronics_7, validation_Energy_and_Power_4, validation_Energy_and_Power_8, validation_Energy_and_Power_9, validation_Materials_22, validation_Math_16, validation_Music_23. 손실 ID는 없음. 채점 보류는 1건 (validation_Energy_and_Power_13).

기존 로컬 Judge와 고정 규칙을 사용했다. 채점에서 보류된 응답은 임의로 판정하지 않았다. 실험은 과거 미추출 40문항과 단일 seed에 한정된다. 모델 생성량 절감은 입력 재처리와 생성·채점의 총비용 절감과 별개다.

[재현 설정](followup_config.json) · [집계 JSON](summary.json) · [실행 코드](nowait_followup.py) · [실행 스크립트](run-followup.sh)

## 채점 보류 기록

- refined_mask · validation_Architecture_and_Engineering_26 · ValueError: Invalid option · terminal_rules_v1:no_unambiguous_terminal_answer
- answer_switch · validation_Energy_and_Power_13 · ValueError: Invalid option · terminal_rules_v1:no_unambiguous_terminal_answer
- refined_mask · validation_Materials_9 · ValueError: Invalid option · terminal_rules_v1:no_unambiguous_terminal_answer
