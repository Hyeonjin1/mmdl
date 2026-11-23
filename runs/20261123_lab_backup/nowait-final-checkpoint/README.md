# NoWait 40문항 후속 실행용 기준 체크포인트

이 아카이브는 객관식 NoWait의 최종 기준 출력이다. 생성 레코드 **40건**, 판정 레코드 **38건**이 있다. 판정 보류 2건은 `validation_Architecture_and_Engineering_26`, `validation_Materials_9`다. 후속 실행기가 읽는 JSONL은 실제 실행 파일의 바이트를 그대로 보존한다. 동결된 `run_config.json`, `target_ids.json`, `blocked_tokens.json`, `cpu_validation.json`도 들어 있다.

초기 5쌍 백업은 중간 스냅샷이다. 이 아카이브는 후속 코드가 참조하는 기존 **NoWait 조건**을 복원한다. 재생성 Baseline, 주관식, 정제 마스크, 답안 전환의 생성 원문은 포함되어 있지 않다.

## 복원과 검증

저장소 루트에서 실행한다. 후속 실행기의 `SOURCE`는 컨테이너의 `/work/results/mmmu_nowait_pilot_20261123`이다. 아래 명령은 그 경로가 비어 있는 새 작업공간에 복원한다.

```bash
checkpoint="$PWD/runs/20261123_lab_backup/nowait-final-checkpoint"
(cd "$checkpoint" && sha256sum -c SHA256SUMS)

target=/work/results/mmmu_nowait_pilot_20261123
mkdir -p -- "$(dirname -- "$target")"
mkdir -- "$target"
tar -xzf "$checkpoint/nowait-reference.tar.gz" -C "$target"
cd "$target"
sha256sum -c "$checkpoint/RECORDS.sha256"
```

두 JSONL의 SHA-256은 [후속 실행 계약](../nowait-followup/followup_config.json)의 `source_generations_sha256`, `source_judgments_sha256`과 일치한다. [후속 실행기](../nowait-followup/nowait_followup.py)는 이 체크포인트 외에 고정된 팀 checkout, 모델과 이미지 캐시, Python 환경을 사용한다. 해당 입력 복원과 환경은 [실습 백업 설명](../README.md)과 [고정 설정](../nowait-snapshot/run_config.json)을 참고한다.
