# Qwen3-VL-4B MMMU-Pro 첫 파이프라인 검증

작성: 2026-11-11T14:41:35. 상태: 소규모 전체 체인 완료.
이번 점수는 소규모 파이프라인 검증 결과이며 성능 개선의 증거 또는 전체 MMMU-Pro 점수가 아니다. 기존 E/L/진단/Judge 실험과 파일은 변경하지 않았다.

## 실제 환경과 모델

호스트 AIX408-4, user, Ubuntu 22.04.5, RTX 4090 24GB, driver 535.274.02. 시작 시 GPU는 Xorg만 18MiB, compute 작업 없음, 디스크 약 48GiB 여유였다. 별도 overlay `.venv-mmpro-first`에서 Python 3.12.12, torch 2.7.0+cu118, Transformers 4.57.1, PEFT 0.17.1, PyArrow 18.1.0을 사용했다. 기존 패키지/드라이버를 변경하지 않았으며 PyArrow만 overlay에 추가했다. BF16 CUDA tensor와 실제 이미지 forward/backward를 통과했다.

FlashAttention2 실제 로드는 `flash_attn` 패키지 부재 ImportError로 실패했다. 실제 모든 단계의 backend는 **SDPA**다. preflight는 optimizer 0 update이고 그 adapter는 본 학습에 사용하지 않았다. 모델/processor/tokenizer는 Qwen/Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. 기존 `/dev/shm/mmdl-user/<revision>` 캐시의 manifest 12개 파일 SHA256/크기를 재검증했다. RAM 캐시는 재부팅 시 사라질 수 있다. 모델 추가 다운로드는 없었다.

실행 시작 base commit: `3a4c0edd0adde42cb05a802d20ebedf76e6e895b`. 새 실행 코드는 당시 미커밋 상태였고 단계별 receipt에 실제 코드 SHA256을 보존했다. 게시 commit은 최종 응답에 별도로 기록한다.

## 데이터와 제외

정규화 pool: {"ArxivQA": 16, "MMFineReason": 16, "OpenMMReasoner": 16, "arxiv_dev": 4, "MMMU_Pro_cached": 64}. 실제 processor 검증 통과: {"arxiv_train": 16, "arxiv_dev": 4, "fine_train": 13, "openmm_train": 16}.

- ArxivQA pin `85a6dca0e2bdc6f0268ae519be8913f83a83cafd`: 전체 annotation JSONL 68,073,816바이트, 이미지 archive 7,948,549,045바이트 중 필요한 이미지까지 스트리밍했다. 전체 8GB 이미지 다운로드 완료로 주장하지 않는다. paper ID version을 제거해 seed42로 paper group 약80/20; 80% 안에서 5% group을 dev로 분리했다. 전체 split IDs는 로컬 `paper_splits.json`에 있다. 획득된 smoke 이미지는 archive 순서로 결정되어 대표성이 없다. 외부20% test는 평가/선정/학습에 사용하지 않았다.
- MMFineReason pin `cda6140bfba449bef9e57065a528b7424966e144`: 첫 Parquet shard와 내장 실제 이미지, `qwen3vl_235b_thinking_response`, answer, source, consistency/pass_rate를 확인했다. consistency=True여도 teacher 최종 답과 annotation이 다르면 제외했다. 원문 reasoning을 사용하고 새 풀이를 생성하지 않았다. inline 객관식 choices는 실제 범위를 추출하고 annotation key와 대조했다. 난이도/일관성 메타데이터는 확인했으나 첫 실험에서 별도의 v1 난이도 선별 효과를 검증했다고 주장하지 않는다.
- OpenMMReasoner pin `c44cf46e41bd24351ae7e7771a44fcd2dbf75f0a`: 5개 config messages/schema/image 경로를 조사하고 `OpenVLThinker-sft-iter3` 6,603,705바이트 Parquet에서 표본을 골랐다. ZIP 두 개의 중앙 디렉터리와 필요한 이미지 member만 Range로 취득했다. 874K 전체 다운로드가 아니다. 이 config는 별도 gold 필드가 없어 공개 assistant answer의 형식/실제 선택지 대응을 검증했다. 독립적인 정답 검증으로 과장하지 않는다.
- MMMU-Pro pin `0d7426df4ccb3d8704a992fe6850abee7c262127`: standard10 첫 shard 346,370,445바이트, vision 첫 shard 430,326,271바이트 및 각각 첫32행 실제 이미지를 캐시했다. 이 cache는 여러 과목이 섞인 Parquet prefix이며 균형 표본이 아니다. 각 setting 2문항을 seed42로 사전 고정했으며 설정 선택에 사용하지 않았다. standard는 질문의 image 위치/순서를 유지하고 실제 선택지만 전달, vision은 공식 screenshot과 공통 지시문만 전달했다. GT와 explanation은 입력 builder에 전달하지 않았다.

실제 행/고유 문항/이미지 수는 pool_counts.json에 구분했다. OpenMMReasoner 16행은 고유2문항·2이미지에 각각8개 공개 teacher 풀이가 붙은 것이다. 이를16개 독립 문항으로 해석하지 않는다. Mixed 실제6회 노출은 Arxiv2/Fine2/OMM2이며 고유5문항이다. 이미지 표본은 Arxiv 그래프, MMFineReason의 공통물성, OMM의 곡선면적 사례를 직접 보았으며 전체 rationale의 수학적 정확성을 전수 인증한 것은 아니다.

원본 source별 파일 SHA256/분할 hash/범위는 metrics.json 및 로컬 download receipt에 있다. 데이터 제외 집계: `{"fine_teacher_final_not_exact_annotation": 41, "fine_reason_too_long_no_truncation": 23}`. 추가 processor 제외3건은 검증 가능한 실제 선택지 없이 단일 문자 정답인2건과 reasoning 내부에 answer 제어태그가 있어 마스크 검사를 통과하지 못한1건이다. 원문이나 정답을 고치지 않고 제외했다.

중복 검사는 전체 ArxivQA dev/test annotation의 정규화 질문, 확인 가능한 paper provenance, 획득한 dev 이미지와 캐시64개 Pro 이미지의 decoded PNG SHA256/질문에 한정했다. MMMU 출처 증류 문항은 제외했다. 확보하지 않은 outer20% 이미지 및 전체 Pro 이미지/문항, perceptual 중복까지 검증했다고 주장하지 않는다. 정확한 중복 제외/범위는 manifest.dedup_scope에 보존했다. 공식 test를 학습에 넣지 않았다.

초기 데이터 서버 502와 만료된 signed image URL 403 실패를 각각 보존했고, 동일 pin의 직접 Parquet 내장 이미지로 전환해 해결했다. 실패를 성공 샘플로 채우거나 정답을 바꾸지 않았다.

## 실제 학습 조건

BF16, language attention/MLP LoRA r8/alpha16/dropout0.05, vision/merger 동결, AdamW LR2e-5/weight_decay0.01/clip1, constant LR, seed42, microbatch1, SFT accum3/preference accum2, gradient checkpointing nonreentrant, no packing. 이미지 min200704/max401408 pixels, 학습 최대3072 tokens, 초과 행 제외, prompt/image/padding 마스킹. Source당 최대16개 pool을 이번 검증에 사용했으며 사용자 상한128 이하다.

공식 processor/chat template의 실제 token을 검사했다. label 앞 `>`와 문자가 합쳐지는 tokenizer 경계 문제 때문에 `<answer>\nB\n</answer>`처럼 태그 안 공백을 사용했다. 내용은 strip 후 uppercase letter 하나다. reasoning 본문/answer 본문/format+EOS를 분리하고 각각 샘플별 유효 토큰 평균 후 `(reason+3*answer+format)/5`, accumulation으로 샘플 평균을 계산했다. 별도 answer1 config는 같은 데이터/순서/step에 계수만1, denominator5를 유지하는 선택적 ablation이며 이번에 실행하지 않았다. next-token shift와 padding, 실제 image backward 및 독립 검사12개를 통과했다.

| 단계 | optimizer update | 노출 | 고유 문항 | 학습 시간(s) | peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| arxiv_sft | 2 | 6 | 4 | 2.5 | 8.76 |
| mixed_sft | 2 | 6 | 5 | 2.3 | 9.90 |
| dpo | 2 | 4 | 4 | 4.4 | 8.97 |
| ranking | 2 | 4 | 4 | 2.9 | 8.68 |

Arxiv-only와 mixed SFT는 원본에서 각각 독립 초기화했다. Mixed는 source별 동수 노출이다. DPO/ranking은 동일한 mixed adapter에서 각각 독립 시작했다. 동결 mixed가 생성한 reasoning 뒤 같은 `</think>\n<answer>\n` prefix에서 실제 모든 라벨의 조건부 token log-prob **합**을 계산했다. 정답 제외 상위2개로 동일 문항/candidate를 구성했다. DPO beta0.1, chosen/rejected completion 전체(닫는 태그/EOS 포함) reference 합을 캐시하고 재계산했다. cache key는 문항/adapter/processor/template/실제 token/mask/pixel/image grid hash를 포함한다. DPO는 문항 내 오답 pair 평균을 낸 뒤 문항 평균; ranking은 동일3후보 label 합 점수의 logsumexp−gold다. 정답 reasoning을 채굴 입력에 주입하지 않았다. 전체4문항×3후보=12개 cache contract의 input IDs/mask/label/pixel/grid/processor hash를 별도 CPU 프로세스로 다시 구성해 모두 일치함을 확인했다. 이는 학습 전 GPU reference 재계산과 별도 검사다. GRPO는 실행하지 않았다.

채굴 결과: `{"valid_questions": 4, "pairs": 8, "failed_questions": 4, "seconds": 27.508169412612915, "reference_frozen_sha256": "803b0274696b6268793533e94bb5e478bafcfba8124a30c595c7cf430666e3a2", "cache_recomputed_pairs": 6, "reference_scope": "entire completion suffix including close tag and EOS; ranking label tokens only"}`.

## 고정 ArxivQA dev 비교

각 모델은 별도 프로세스에서 reload하고 생성했다. Free 생성과 reasoning-prefill `<think>` 뒤 label likelihood는 별도의 생성 두 번이므로 같은 reasoning을 공유한 한 번의 평가라고 해석하지 않는다. 둘 다 GT/rationale 없이 생성했다. greedy, 자유384/추론512 새 토큰 상한. 형식 실패/길이 종료는 분모에 남고 이 검증의 정확도에서 오답이다. margin은 완결된 reasoning에서 gold−최상위nongold label log-prob 합 평균이며 실패 문항은 계산 불가로 남는다.

| 모델 | 자유 answer-tag 정확도 | two-block 엄격 정확도 | reasoning 뒤 label LL 정확도 | 포맷 준수 | 자유/추론 길이 종료 수 | 평균 margin |
|---|---:|---:|---:|---:|---:|---:|
| initial | 3/4 (75.0%) | 0/4 | 1/4 (25.0%) | 0/4 | 0/0 | 11.374999761649633 |
| arxiv_sft | 3/4 (75.0%) | 0/4 | 3/4 (75.0%) | 0/4 | 0/0 | 9.791666761734026 |
| mixed_sft | 3/4 (75.0%) | 0/4 | 2/4 (50.0%) | 0/4 | 0/0 | 4.791666567670745 |
| dpo | 3/4 (75.0%) | 0/4 | 1/4 (25.0%) | 0/4 | 0/0 | -1.5624997615850589 |
| ranking | 3/4 (75.0%) | 0/4 | 2/4 (50.0%) | 0/4 | 0/0 | 11.249999644905301 |

중요한 형식 실패: 모든 모델이 자유 생성에서 answer 태그는 출력했지만 think 태그를 생략했다. answer-tag 열은 유일한 유효 answer 블록만 별도 추출한 보조 점수이며, 엄격 two-block 점수/원래 로그는 변경하지 않았다. 전20개 dev 출력에서 길이 종료0, 포맷준수0이므로 answer-tag3/4를 완전한 지시문 준수나 학습 성공의 증거로 보지 않는다. 두 점수 모두 실패 문항을 제외하지 않고4를 분모로 쓴다.

LL 점수를 계산할 수 있었던 완결 reasoning 수는 초기1/4, Arxiv SFT3/4, 혼합3/4, DPO2/4, ranking2/4다. 나머지는 EOS까지 생성했지만 유효한 비어 있지 않은 think 블록이 없어 추출 실패로 분모에 남겼다. 따라서 LL 정확도 차이는 reasoning 형식의 완결 여부 변화도 포함하며 추론 능력의 순수한 변화가 아니다. margin 평균도 점수를 계산한 문항에서만 구했으므로 모델 간 평균의 표본이 다르다. generation_audit.json에 별도 집계했다.

모델 선택: `arxiv_sft`. Arxiv dev 자유 정확도 → LL → 포맷 → 사전 모델 순서 tie-break로 고정했다. MMMU-Pro 결과는 선택에 사용하지 않았다.

## MMMU-Pro 입력/채점 점검

| 설정 | n | 공식 parser 정답 | 엄격한 two-tag 정답 | 자체 label LL 정답 | 실행 오류 |
|---|---:|---:|---:|---:|---:|
| MMMU_Pro_standard10 | 2 | 1 | 0 | 0 | 0 |
| MMMU_Pro_vision | 2 | 1 | 0 | 0 | 0 |

공식 repository parser commit `268471d0d488258990025331c7528359c324aa25`의 함수를 그대로 사용했으며 fallback random에 seed42를 고정했다. 공통 지시문은 사용자가 지정한 two-tag 형식이므로 공식 Answer:$LETTER 프롬프트를 그대로 재현한 leaderboard 점수가 아니다. 공식 parser는 reasoning의 option 언급이나 random fallback을 인정할 수 있어 엄격한 최종 answer와 다르다. MMMU-Pro4개 자유 출력도 think 태그를 생략했다. 별도 answer-tag 추출은 standard1/2, vision0/2이며, 공식 parser는 vision 한 문항의 잘못된 최종 answer 대신 reasoning 중간의 선택지를 채택해1/2를 부여했다. 이 불일치는 원문으로 확인했으며 공식 parser 점수를 최종 답 정확도와 동일시하지 않는다. 전체3460문항 평가를 실행하지 않았다. 이 n=2 점수로 MMMU-Pro 개선을 주장할 수 없다.

## 저장 위치와 다음 실행

- 실제 source/normalized data: `datasets/raw/MMProFirst_20261111`, `datasets/processed/MMProFirst_20261111`.
- 코드: `code/mmpro_first/`; 재현 config: `configs/MMProFirst_v1/`.
- 단계별 adapter: `runs/MMProFirst_20261111/{arxiv_sft,mixed_sft,dpo,ranking}/adapter/`.
- loss/원문 예측/reference cache/실패/receipt: 같은 run 및 `.restore/mmpro_first_v1/`. 원문과 가중치는 공개 Git에 넣지 않았다.
- 공개 문항 ID·정오판정: `question_decisions.csv`; 데이터 ID·이미지/타깃 hash·토큰 수: `sample_manifest.csv`.

전체 재실행 명령은 코드 README에 있다. `launch.py prepare --run-id MMProFirst_expanded_01 --profile expanded` → `launch.py train --run-id MMProFirst_expanded_01`은 source당 최대128개 확장 pool, SFT128 update 및 최대32개 채굴/16 preference update의 별도 실행이다. 이는 123K/874K 전량 학습이 아니다. dev에서 설정을 잠근 뒤 `launch.py final --run-id MMProFirst_expanded_01`을 별도로 실행하면 outer20% ArxivQA와 full standard10/vision을 평가한다. 이 장시간 명령들은 제공만 했고 이번 소규모 실행에 포함하지 않았다. 전체 규모에서 실행 검증되었다고 주장하지 않는다. 추가 설정/epoch/GRPO를 자동 실행하지 않는다.
