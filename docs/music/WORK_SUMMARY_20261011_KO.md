# Music 작업물 핵심 정리

정리 기준일: 2026-10-11 (KST). 대상: 로컬 `Music/`의 프로젝트 문서, 실험 기록, 집계 지표 및 Git 이력. 원격 서버의 현재 실행 상태를 새로 조회한 보고서는 아니다. 근거 파일의 상대 경로와 SHA256은 [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json)에 기록했다.

## 핵심 결론

주된 연구 흐름은 **OneRef+SAM2의 텍스트 기반 마스크 예측·보정 → BindRef의 객체 후보 생성과 언어 선택 분리 → 후보 경쟁을 고려한 단서 기반 학습**이다. 같은 프로젝트의 실험별 체크아웃이 많으므로 폴더 개수를 독립적인 연구 수로 해석하지 않는다.

- BindRef는 후보 품질, 선택 정확도, 최종 마스크 품질을 분리해 측정한다. Qwen8B의 전체 RefCOCOg val SAM2 refinement 기록은 원본 해상도 mIoU **79.60%**, cIoU **78.49%**다. 최근 M2F/Qwen4B SFT 기록은 원본 selected-mask mIoU **84.8863%**, cIoU **85.7692%**다. 후보 생성기·입력·학습 조건이 달라 두 기록을 단순한 개선 전후로 비교할 수 없다.
- 후보 경쟁을 줄이거나 목표를 상기시키면 일부 오류가 복구된다. 그러나 attention 증가 자체가 선택 개선을 보장하지 않으며, 일반적인 GT-free 동적 단서 조절 방법은 아직 검증되지 않았다.
- 최신 Stage-2 GRPO는 단서 보상, 경쟁 후보 풀 구성, 두 요소 결합을 같은 부분 train에서 비교하도록 준비되었고, 최신 로컬 기록에 pilot 통과와 본 학습 시작이 남아 있다. 최종 전체 검증 성능은 이 보고서에서 확인되지 않았다.
- other-target 판별은 multiview에서 오류 검출 recall이 높아졌지만 오탐도 늘었다. 모델과 프롬프트가 함께 달라 순수한 입력 효과로 단정할 수 없다.
- Track2는 이미지 RES 모델을 MeViSv2로 옮긴 구현과 5개 epoch 기록이 있다. 남아 있는 검증은 최대 1,024개 frame sample 설정이며 공식 전체 영상 평가·제출 성적으로 제시하지 않는다.

## 1. 프로젝트 구성

| 작업군 | 폴더 | 핵심 작업 |
|---|---|---|
| BindRef 기본·인지 진단 | `bindref`, `bindref_rsim_20260928`, `bindref_gemma_interventions_20260928`, `bindref_attention_budget_20260929` | 텍스트 없이 객체 후보 생성, LVLM 선택, 입력 격리·SSI·모델별 진단 |
| M2F·선택 학습 | `bindref_m2f_score015_auto_20260930`, `bindref_pairwise_m2f_20261004` | Mask2Former 후보, pairwise/ranking 및 candidate attention gate 실험 |
| Qwen SFT·GRPO | `bindref_qwen4b_grpo_baseline`, `bindref_qwen4b_grpo_dual`, `bindref_qwen8b_resume`, `bindref_qwen8b_resume_baseline` | baseline, 혼합 보상·attention 지도, 저장 adapter 이어 학습 |
| 동적 단서·최신 실행 | `bindref_stage2_dynamic`, `bindref_stage2_cue_baseline`, `bindref_stage2_cue_grpo`, `bindref_attention_runtime_20261011` | 단서–후보 표, 후보 풀별 보상·경쟁자 구성, 실행·오프라인 runtime·보관 도구 |
| OneRef+SAM2 | `sam2_oneref`와 `sam2_oneref_*` 5개 사본 | reliability CGPP, safe negative, mask prior, residual repair, EACS/L-CAR/InternVL/teacher point ablation |
| 오류 판별 | `qwen_other_target_identity_*` 3개 폴더 | 전체 장면·crop·multiview의 SAME/OTHER 판별 |
| 영상 확장 | `track2` | MeViSv2 loader, frame 학습, 존재 여부 분류, SAM2 memory 전파 설계 |
| 공용 자산·전송 | `datasets`, `models`, `slot_cache`, `envs`, `venvs`, `transfers`, `transfer_*`, bundle/tar.xz | 데이터·가중치·캐시·환경·이동용 스냅샷; 독립 성능 결과로 계산하지 않음 |

`bindref_*`의 origin 상당수는 로컬 `.bundle`이다. 이는 실험 전달용 Git 스냅샷이며 GitHub 업로드 대상으로 해석하지 않았다. 이번 정리는 기존 보고서 저장소 `Hyeonjin1/mmdl`에 추가한다.

## 2. BindRef: 후보 생성과 선택 분리

파이프라인: 이미지 → visual encoder/instance slot head → 후보 mask·품질·공간 정보 → LVLM의 referring expression 기반 slot 선택 → 선택적 SAM2 refinement.

후보 생성은 query text 없이 수행한다. SAM2 Hiera-L, DINOv3-B/H+, Co-DINO ViT-L을 같은 인터페이스로 비교하며, 최근 선택 학습에는 M2F 후보도 사용한다. 후보 oracle upper bound를 먼저 측정해야 selector의 오류와 후보 누락을 구별할 수 있다. GT mask, oracle 번호와 target IoU는 추론 prompt에 넣지 않는다.

### 기록된 대표 성능

| 설정·근거 | 평가 범위 | 지표 | 기록값 |
|---|---|---|---:|
| Qwen3-VL-8B 4-bit language-only QLoRA, DINOv3-H+ slot, best 0.6 epoch | Full val 1,300 image group / 4,896 expression | selected-slot cached 256 proxy mIoU / cIoU | 79.09% / 77.82% |
| 같은 8B selector + epoch-11 SAM2 refinement | 같은 full val | 원본 refined mIoU / cIoU | 79.60% / 78.49% |
| 같은 8B 설정 | 같은 full val | exact oracle-slot accuracy / 후보 oracle proxy mIoU | 88.62% / 84.27% |
| M2F E1 score 0.015, NMS 0.75, Qwen4B SepView SFT r64/alpha128 | Full val 1,300 group / 4,896 query; 최신 실험의 baseline 기록 | 원본 selected-mask mIoU / cIoU / exact | 84.8863% / 85.7692% / 93.9747% |
| 최신 M2F GRPO 입력의 후보 상한 | 같은 full val | 원본 oracle mIoU | 87.8205% |

8B 수치는 `docs/experiments/CURRENT_EXPERIMENTS.md`, 4B 수치는 최신 Stage-2 GRPO 실험 메모에서 인용했다. 이번 정리에서 추론을 다시 실행하지 않았다. 4B full-train IoU GRPO cIoU **86.3886%**도 최신 메모의 참조값으로 존재하지만, 해당 종료 집계 파일을 이 보고서에서 독립 검증하지 않았다. Stage-2 부분 train과의 matched control로 사용하지 않는다.

Full val, 561개 고정 진단 코호트, Limit Val(377 image group / 1,400 expression), 27개 demo, 33개 pilot을 구분한다. proxy mask 지표와 원본 해상도 지표를 섞어 증가량을 만들지 않는다. exact accuracy는 후보 identity가 바뀌면 직접 비교할 수 없다.

## 3. 후보 경쟁·단서·attention에서 얻은 근거

### 상향식 입력 격리와 SSI

SepView는 후보별 시각 입력을 분리해 경쟁 간섭을 줄이려는 구성이다. Co-DINO 561 진단의 유효 후보 4,318개에서 Tiled→SepView 자기 객체 ViT attention 평균이 **25.732%→35.004%**로 증가했다. 모든 후보에 나타난 변화이며, 정답에만 선택적으로 주는 신호나 선택 성능 측정이 아니다.

현재 독립 Yes/No SSI의 공통 기준은 crop 없는 전체 장면, 단일 빨간 윤곽선, 이미지 밖 Slot 지정이다. 기준 평균 Yes 반응은 **0.833515**, SSI는 기본 SoM **0.160415**, SoM+후보 패널 **0.209096**, SepView **0.140928**, Tiled **0.216168**로 기록된다. 생물학적 억제의 직접 측정이 아니라 출력 반응의 조작적 진단이며, 과거 crop 기준 SSI와 혼용하지 않는다.

### 목표 상기와 attention 개입

| 진단 | 결과 | 해석 |
|---|---|---|
| 원래 오답 대상 목표 상기 2턴 vs 무관한 대화 2턴 | zero-shot 557오답 복구 17.8% vs 5.7%; tuned 120오답 13.3% vs 4.2% | 오답 코호트에서 긍정적 복구 근거. 전체 val의 순이득·기존 정답 손상은 미입증 |
| SepView 561에서 원문 재강조 / 현재 후보 기반 cue 강조 | baseline 456/561, 각각 449/561 / 448/561 | 이 구성에서는 전체 코호트 개선 없음 |
| GT 후보 answer attention bias +4, Co-DINO 561 | 454/561→510/561, 57복구/1손상 | GT를 아는 oracle 진단; GT-free 방법 성능 아님 |
| M2F 561의 오답 94개에서 선택된 경쟁자 제거 vs 대조 후보 제거 | 64/94복구 vs 12/94복구 | 경쟁자 영향 근거. 후보 제거에 따른 컨텍스트 변화도 남음 |
| 경쟁자 제거 시 cue attention 변화 패턴을 원래 풀에 적용 | fresh baseline 51오답 중 0복구, 2정답 중 1손상 | 해당 transplant 연산은 유효한 복구 방법이 되지 못함 |
| GT cue→GT 후보 attention 강화, layer 14/21/28 × bias 0.5/2 | 6×561 조건 완비; 실제 attention 증가에도 L14 1복구, L21 0복구, L28 1손상 | attention 증가만으로 선택 개선을 보장할 수 없음 |

핵심 미해결 문제는 원래 목표를 유지하면서 **현재 후보 풀에서 필요한 단서–후보 연결을 GT 없이 얻고, 전체 val 원본 cIoU 개선으로 전달하는 것**이다. 관찰 attention이 높다는 사실만으로 올바른 의미 결합이나 인과적 개입 성공을 주장하지 않는다.

## 4. 최신 Stage-2 cue GRPO

최신 근거는 `bindref_attention_runtime_20261011`의 2026-10-11 메모다. 일부 메모 첫 줄에는 setup pending이 남아 있지만, 후속 항목에는 pilot 통과·본 학습 재초기화·process/log/GPU 확인이 기록되어 있어 후속 항목을 상태 해석에 반영했다.

| 변형 | 후보 풀 | 보상 | 최신 로컬 기록상 상태 |
|---|---|---|---|
| reward | 기존 random full/partial | 0.8 native grouped IoU + 0.2 cue satisfaction | pilot 통과, 본 학습 시작; 최종 archive 대기 |
| pool | cue-guided competitors | 기존 IoU only | pilot 통과, 본 학습 진행 기록; 최종 archive 대기 |
| combined | cue-guided competitors | 혼합 cue 보상 | pilot 102 train query·3 optimizer update·33-query smoke val 통과, 본 학습 원 SFT에서 재시작; 최종 archive 대기 |

공통 train은 **8,503 group / 27,567 query**로 원래 80,485 query의 **34.2511%**다. Stage-2 표가 확보된 샘플을 골랐으므로 무작위 부분집합이 아니다. 세 변형은 동일 subset·순서를 사용하고, val은 4,896 query를 유지하며 0.1 epoch마다 평가하도록 설정했다. LR 1e-6, G=2, temperature 0.8, seed 42, accumulation 8, 원 SFT adapter에서 시작하는 새 optimizer다.

단서 표의 GT oracle 행을 all-Y로 보정한 teacher를 사용하며 **7,107 sample / 10,649 cell** 변경을 기록했다. 이는 학습용 표 보정이고 의미적 정확성을 독립적으로 보증하지 않는다. 표와 GT 필드는 추론 prompt에서 제외한다. U/NV·scene-only·비식별 단서는 보상에서 제외하고, 유효한 단서가 없으면 기존 IoU 보상을 사용한다.

561 val 진단 표의 교차 단서 pool 구성은 구조적으로 129개, 엄격한 visual/spatial 조건에서는 25개가 가능했다. 엄격한 25개 중 사람 승인 표는 0개였다는 별도 audit가 있다. 이 결과는 데이터 구조의 가능성 확인이며 검수 완료·학습 효과·val 일반화 증거가 아니다.

최신 runtime 체크아웃에는 attention-best adapter를 초기값으로 사용하는 후속 IoU GRPO qualification·수집 코드도 있다. Git 코드 존재만으로 해당 실험 완료나 성능 향상을 판정하지 않았다.

## 5. OneRef+SAM2: 이전 연구 기반

OneRef grounding mask에서 reliability CGPP point prompt를 만들고 SAM2가 마스크를 보정한다. safe negative, mask prior, 학습 가능한 prompt policy, residual repair, train-time LVLM critic 등의 ablation이 축적되어 있다. 파생 폴더는 특정 commit의 재현·구조 비교용 사본이다.

| 근거 기록 | 평가 | 수치·상태 |
|---|---|---|
| `sam2_oneref/notes/experiment_results.md`의 baseline | RefCOCOg val 학습 validation | SAM2 mIoU 75.89% |
| native preference, SAM2 freeze / OneRef만 학습 | 같은 문서의 학습 validation | SAM2 mIoU 76.44%, baseline 대비 +0.55%p; 10 epoch 완료 기록 |
| 같은 native-preference best의 standalone 평가 | val 4,896 sentence | mIoU 76.70%, cIoU 74.36%, empty prediction 0 |
| 보존된 safe-negative + mask-prior run | best epoch 5 validation | SAM2 mIoU 77.0132%, OneRef mIoU 66.0435% |
| native contrast critic-guided actor | 두 설정 validation | SAM2 mIoU 74.82% / 75.64%, 해당 baseline보다 하락 |

오래된 `EXPERIMENTS.md`의 2026-04-19 running 표나 `EXPERIMENT_RESULTS.md`의 2026-05-18 상태는 당시 기록이다. 지금도 실행 중이라고 해석하지 않는다. 위 결과는 이 계열의 대표 근거이며 모든 후속 실험의 전역 최고값을 전수 조사한 표는 아니다.

## 6. other-target identity 판별

집계 JSON의 공통 primary scope는 `normal vs other_ref + other_unref_ann` **4,860개**이고, `other_unann` 36개를 제외한다. 4,896개 전체 행을 포함한 별도 집계를 섞지 않았다. 아래 양성은 OTHER 오류다.

| 입력 / 모델 | Accuracy | OTHER precision | OTHER recall | FPR | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| 전체 장면 / Qwen3-VL-30B-A3B Q4 | 84.01% | 20.37% | 24.45% | 9.85% | 57.30% |
| crop / 같은 30B-A3B Q4 | 85.39% | 23.98% | 25.99% | 8.49% | 58.75% |
| multiview / Qwen3-VL-32B Q4 | 82.59% | 30.16% | 65.64% | 15.66% | 74.99% |
| multiview conservative / 같은 32B Q4 | 83.99% | 27.12% | 42.29% | 11.71% | 65.29% |

multiview 32B는 놓치는 OTHER 오류를 줄이는 대신 정상 사례 오탐이 늘었다. conservative prompt는 오탐을 낮추면서 recall도 낮췄다. 대부분이 정상인 분포라 accuracy만 보면 검출 능력을 과대평가할 수 있다. 32개 smoke 결과는 위 full-scope 표에서 제외했다.

## 7. Track2 / MeViSv2

OneRef+SAM2 RES를 referring video object segmentation으로 옮기는 작업이다. MeViSv2 native loader, frame-level trainer, 존재 여부 분류, reliability CGPP·mask prior·full unfreeze가 구현되어 있다. 설계 문서는 keyframe seed와 SAM2 forward/backward memory propagation, track 선택·앙상블을 제안한다. 문서상 계획과 실제 평가 완료를 구분한다.

남은 출력 `mevisv2_track2_exist_balanced_b2_side768_20260718_154825`의 `metric.json`은 **epoch 0–4**를 담고 있다. 설정은 train/valid_u, frame stride 5, side 768, max val sample 1,024, val no-target sample 256이다. 이 실행의 `iou_oracle_weight`는 **0**이라 문서상 IoU reward repair 계획이 실제 켜졌다고 해석하지 않는다.

| 기록 | J | F | J&F | T accuracy | N accuracy | 로컬 `metric_track2_final` |
|---|---:|---:|---:|---:|---:|---:|
| best epoch 0 | 54.75% | 25.96% | 40.36% | 96.09% | 35.94% | 57.46% |
| last recorded epoch 4 | 52.33% | 25.17% | 38.75% | 95.31% | 33.98% | 56.02% |

로컬 지표상 best가 첫 epoch에 남고, no-target 정확도가 낮다. 개선 후보는 경계 품질과 존재 여부 판별이지만, 전체 영상 memory inference·공식 평가·제출 성공은 이 기록만으로 확인되지 않는다.

## 8. 다음 확인 지점과 보관 범위

1. 최신 세 Stage-2 GRPO의 최종 archive와 4,896-query coverage, best/last metric을 수집해 실제 결과를 갱신한다. 부분 train과 full-train GRPO 차이는 matched control 없이는 인과적 개선으로 부르지 않는다.
2. 후보 생성기·NMS·입력·train subset·해상도를 고정한 비교를 구성하고, selected-mask native mIoU/cIoU와 exact accuracy를 함께 기록한다.
3. attention 개입은 attention 변화뿐 아니라 답변 logits, 복구/손상, 최종 마스크 지표까지 확인한다. 별도의 competitor-swap 평가로 단서 기반 기제를 검증한다.
4. Track2는 현재 frame subset 성적과 별도로 전체 영상 전파·공식 metric 평가를 확인한다.

이번 업로드는 핵심 요약과 출처 inventory만 담는다. 데이터셋·이미지·가중치·raw response·대형 로그·가상환경·전송 archive는 포함하지 않는다. 압축된 과거 프로젝트는 풀지 않았으며, 포함된 내용까지 읽었다고 주장하지 않는다. 출처 경로는 `Music/` 상대 경로이고 서버 주소·접속 정보는 요약에 넣지 않았다. 기존 P03/P04 보고서는 별도 연구 기록으로 유지한다.
