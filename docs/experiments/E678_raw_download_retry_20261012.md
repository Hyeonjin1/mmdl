# E6·E7·E8 원본 다운로드 재시도 — 2026-10-12

재시도 결과: 전부 실패, 완료 항목 없음. sam2_oneref.zip 삭제 후 현재 파일 시스템 여유는 약 40GiB다. 이번 실패는 디스크 부족이 아니라 실행 환경에서 배포 호스트의 DNS 이름을 해석하지 못한 것이다. 호스트 전체의 네트워크 고장 여부는 이 결과만으로 확정하지 않는다.

이전 HEAD 사전 검사와 달리 이번에는 원본 8개 모두 실제 GET 다운로드 요청으로 재시도했다. curl --continue-at - 로 .partial 파일에 재개하도록 요청했지만 모두 exit code 6 / Could not resolve host 로 종료됐다. 기존 정상 원본은 없었고 새로 받은 원본도 없다. 환경이나 DNS 설정은 변경하지 않았다.

| 묶음 | 실제 저장 절대 경로 | 상태 | 사용량·실제 확보 수 | 정답·풀이·QA 사용 및 한계 |
|---|---|---|---|---|
| 과학 도식·관계·계산: ScienceQA + AI2D | /home/user/mmdl-work/datasets/raw/ScienceQA 및 /home/user/mmdl-work/datasets/raw/AI2D | 실패 | manifest만 저장, 원본 0B, 이미지·문항 0개 | annotation 미확보. 사용 불가. 계산·관계 후보 수 집계 불가; 공학 계산 전용 자료로 분류하지 않음 |
| 투상도·단면도·공간 대응: Fusion 360 Gallery | /home/user/mmdl-work/datasets/raw/Fusion360Gallery | 실패 | manifest만 저장, 원본 0B, CAD·이미지 0개 | reconstruction r1.0.1 요청. 확보 뒤에도 QA 가공 필요 |
| 악보 기호·음높이·음가: PrIMuS | /home/user/mmdl-work/datasets/raw/PrIMuS | 실패 | manifest만 저장, 원본 0B, 이미지·annotation 0개 | 기본 PrIMuS 요청. 전사 annotation을 QA로 간주하지 않음; 추가 가공 필요 |

실제 다운로드 시각·수신 파일 SHA-256·실제 revision은 여전히 null이다. 원본 미확보로 archive 무결성·annotation 읽기·임의 이미지 열기·연결 검증은 수행하지 못했다. 공식 split은 미확보이며 임의 split은 만들지 않았다. 학습·GPU 추론·질문 생성·렌더링·학습 데이터 가공을 실행하지 않았다. 기존 작업을 중단하지 않았다.

최신 요청 URL·시각·실패 내용 및 이전 시도 이력은 각 datasets/raw 폴더의 manifest.json과 [재시도 manifest 사본](E678_raw_download_retry_manifest_20261012.json)에 기록했다. 공식 출처·라이선스·선정 기준은 [최초 감사 보고서](E678_raw_dataset_download_20261011.md)에 있다. 원본·archive는 Git에 포함하지 않는다.

보고서 원격 게시도 DNS 제약에 영향을 받는다. 연결 복구 후 게시 재시도 명령: `git -C /home/user/mmdl-work push origin main`. 다운로드 재개에는 먼저 실행 환경에서 배포 호스트의 DNS/네트워크 접근이 가능해야 한다.

## 일반 터미널에서 실행하는 대안

이 실행 환경에서는 wget으로 Hugging Face를 조회해도 DNS 오류가 발생했다. 실습컴 일반 터미널의 네트워크가 정상이라면 다음 명령으로 다운로드를 실행할 수 있다: `python3 /home/user/mmdl-work/scripts/e678_download_from_terminal.py`. 이 스크립트는 정상 기존 파일을 재사용하고 .partial에 재개 다운로드한다. JSON 읽기, ZIP CRC, tar.gz 읽기와 SHA-256 기록을 수행하며 원본을 보존한다. 8GiB의 디스크 여유를 남기도록 다운로드 크기를 제한한다. 압축 해제·이미지 연결 점검은 별도이며 데이터셋 전체 완료로 표시하지 않는다. 실제 Git revision 해결과 AI2D 별도 공식 test ID 파일 확보도 남아 있다. 새 terminal_download_manifest.json에 기록하며 이전 실패 manifest는 보존한다. 문법 확인만 수행했고 일반 터미널에서 실행한 결과는 아직 없다.

ScienceQA 공식 저장소가 안내하는 브라우저 대안: [Google Drive](https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw), [Hugging Face](https://huggingface.co/datasets/derek-thomas/ScienceQA). 후자는 원본 archive와 포장이 다를 수 있어 파일·ID·split과 lecture/solution의 동일성을 확인하기 전 원본과 동일하다고 확정하지 않는다.
