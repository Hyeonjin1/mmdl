# E6·E7·E8 후보 원본 확보 점검 — 2026-10-11

다운로드 완료 항목은 없다. 실행 환경의 DNS/네트워크 제약으로 모든 요청이 `curl: (6) Could not resolve host`로 실패했다. ScienceQA JSON은 실제 GET을 시도했고 대용량 archive는 다운로드 전 HEAD 단계에서 실패했다. 이는 배포처의 링크 단절을 확인한 결과가 아니다. 웹 문서 열람은 별도 도구에서 가능했으나 원본 파일을 실습컴에 받았다는 뜻은 아니다.

프로젝트 `/home/user/mmdl-work`, 기존 데이터 위치 `/home/user/mmdl-work/datasets/raw`. 시작 시 파일 시스템 가용량은 4,755,181,568 bytes(약 4.43 GiB), 기존 raw는 약 1.1 GiB였다. raw에는 ChartQA_P04, MMMU_latest_vllm, MMMU_restore, PGPS9K_P04, TabMWP_P04가 있었다. 대상 이름으로 기존 파일을 검색했으나 재사용 가능한 자료는 발견하지 못했다. 다른 경로에 이름이 다른 archive가 있는지는 확정하지 못했다.

학습, GPU 추론, 질문 생성, 학습 데이터 가공, 렌더링, 단면 생성, 선택지 제작, 환경 설치·변경, 다른 작업 종료는 수행하지 않았다. sandbox 프로세스 목록만 보였으므로 호스트 E4·E5 실행 상태를 확정하지 못했다. 기존 데이터·실행 파일에는 손대지 않았다.

| 묶음 / 데이터셋 | 실제 저장 절대 경로 | 상태 | 디스크 사용 / 실제 확보 수 | 정답·풀이 annotation | QA 사용 가능성 및 한계 |
|---|---|---|---|---|---|
| 과학 도식·관계·계산 / ScienceQA | /home/user/mmdl-work/datasets/raw/ScienceQA | 실패: JSON GET 및 archive 사전 요청 DNS 오류 | manifest만 약 8 KiB; 원본 0 B; 이미지·문항·CAD 각 0 | 미확보. 공식 문서는 answer, lecture, solution을 설명 | 현재 사용 불가. 원본 QA는 있으나 공학 계산 전용이 아니므로 문항별 선별 필요 |
| 과학 도식·관계·계산 / AI2D | /home/user/mmdl-work/datasets/raw/AI2D | 실패: archive 사전 요청 DNS 오류 | manifest만 약 8 KiB; 원본 0 B; 이미지·문항·CAD 각 0 | 미확보. 공식 안내에 질문·정답·도식 annotation 있음; 풀이 존재는 미검증 | 현재 사용 불가. 도식 이해와 공학 수치 계산을 구분해야 함 |
| 투상도·단면도·공간 대응 / Fusion 360 Gallery | /home/user/mmdl-work/datasets/raw/Fusion360Gallery | 실패: r1.0.1 사전 요청 DNS 오류 | manifest만 약 8 KiB; 원본 0 B; 이미지·문항·CAD 각 0 | QA 정답·풀이 제공 자료로 확인되지 않음 | 현재 사용 불가. 확보 이후에도 QA 가공 필요; 썸네일은 투상도·단면도 정답 세트가 아님 |
| 악보 기호·음높이·음가 / PrIMuS | /home/user/mmdl-work/datasets/raw/PrIMuS | 실패: 기본 archive 사전 요청 DNS 오류 | manifest만 약 8 KiB; 원본 0 B; 이미지·문항·CAD 각 0 | 전사 annotation은 공식 제공 안내에 있으나 미확보; QA 정답·풀이 없음 | 현재 사용 불가. 음정 질문에는 해석·정답 제작 필요; 단선율 자료의 화음 실험 적합성 제한 |

위 디스크 사용은 폴더와 manifest의 할당량이며 데이터 다운로드량이 아니다. 문서의 전체 데이터 수를 실제 확보 수로 사용하지 않았다. archive 검사, annotation 읽기, 임의 이미지 열기, 이미지–annotation 연결 검사 모두 원본 미확보로 수행하지 못했다. 실제 다운로드 시각·파일 크기·SHA-256·실제 revision은 null로 기록했다. 실패 시각과 요청 URL, curl 오류, 요청 배포 버전, 라이선스 안내는 각 manifest에 남겼다.

## 공식 배포 확인과 최소 범위

- [ScienceQA 공식 저장소](https://github.com/lupantech/ScienceQA): problems.json의 질문·선택지·answer·lecture·solution·split을 보존할 계획. [공식 다운로드 스크립트](https://raw.githubusercontent.com/lupantech/ScienceQA/main/tools/download.sh)의 S3 train/val/test.zip을 선정했다. Google Drive와 저자 Hugging Face 배포도 공식 대안으로 안내돼 있다. main은 실제 commit SHA가 아니며 원본을 받기 전 고정 revision을 해결해야 한다. 데이터 라이선스는 CC BY-NC-SA 4.0이며 코드 라이선스와 구분한다.
- [AI2D 공식 PRIOR 안내](https://prior.allenai.org/projects/diagram-understanding): 모든 이미지·질문·annotation 묶음과 별도 Test Set IDs가 범위다. [공식 AWS registry](https://registry.opendata.aws/allenai-diagrams/)가 안내하는 ai2-public-datasets 버킷의 diagrams/ai2d-all.zip 경로는 [Google Research 코드](https://github.com/google-research/pix2struct)에서도 확인했다. 웹 도구가 PRIOR 다운로드 링크를 읽지 못해 해당 링크의 실제 URL과 test ID 파일 URL은 아직 확보하지 못했다. 비공식 미러는 사용하지 않았다. registry는 CC BY-SA를 표시하지만 세부 버전은 원본에서 재확인해야 한다.
- [Fusion 360 공식 README](https://github.com/AutodeskAILab/Fusion360GalleryDataset): reconstruction r1.0.1 약 2.0GB만 선정. assembly, segmentation, 추가 extrude·사전 가공 묶음은 제외. [reconstruction 문서](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/reconstruction.md)는 SMT/STEP, OBJ, JSON, PNG 썸네일 및 train_test.json(6900/1725)을 설명한다. 전체 확보 뒤에도 QA 가공 필요로 분류해야 하며 현재는 다운로드 실패다. [Autodesk 라이선스](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md)는 비상업 연구 용도와 전체 재배포 제한을 명시한다.
- [PrIMuS 공식 페이지](https://grfia.dlsi.ua.es/primus/): 기본 배포의 PNG, semantic, agnostic, MEI, Plaine and Easie 원본을 범위로 잡았다. 기본 archive URL은 웹 도구에서 gzip 응답까지 확인됐으나 실습컴 HEAD는 DNS 오류였다. Camera-PrIMuS는 제외한다. 홈페이지에 공식 split·라이선스의 명확한 안내가 없어 archive 내부를 점검해야 한다. split 없음이라고 확정하지 않았으며 임의 split은 만들지 않았다.

## 계산·관계 문항 예비 집계

ScienceQA 및 AI2D 실제 문항이 없으므로 수치 계산 / 관계식 적용 / 단위 변환 / 도식 관계 해석의 후보 수는 모두 **집계 불가(null)** 이다. 0문항이라고 판정한 것이 아니다.

확보 후 원본을 바꾸지 않는 감사 집계에서 다음 기준을 사용한다. (1) 수치 계산은 질문이 산술·비율·측정값 계산을 요구하고 실제 수치 조건이 있는 경우, (2) 관계식 적용은 제공 조건을 물리·수학 관계식에 대입하거나 변환해야 답할 수 있는 경우, (3) 단위 변환은 명시된 단위 사이의 환산이 답을 결정하는 경우, (4) 도식 관계 해석은 화살표·연결·경로·순서·포함·공간 관계를 읽어야 답하는 경우다. 단순 물체 명칭·기호 식별은 도식 관계와 분리한다. 과목명, 이미지 유무, 숫자 하나, lecture의 일반 공식만으로 분류하지 않는다. 질문·선택지·hint와 원본 solution 및 이미지의 근거를 함께 확인해야 한다. 각 범주는 중복 가능하므로 범주별 수와 합집합 수를 구분하고 불확실한 사례를 별도 기록한다. 현재는 이 기준을 실행하거나 새 질문을 생성하지 않았다.

## 재개 및 게시

`python3 scripts/e678_raw_download_probe.py`는 JSON 다운로드 시 `.partial`에 curl `--continue-at -`를 사용하고 JSON 파싱 성공 후에만 이름을 확정한다. 기존 파일은 덮어쓰지 않는다. archive 요청은 HEAD만 수행하므로 이 스크립트만으로 전체 다운로드·검증이 끝나지 않는다. 압축 원본을 보존하면서 압축 해제 공간을 다시 계산하고 원본 수신을 별도 수행해야 한다. Fusion archive 약 2GB와 AI2D 약 945MB만으로도 현재 여유의 상당 부분을 사용하므로 세 묶음의 동시 저장·해제를 안전하다고 판단할 근거가 없다. 기존 실행의 디스크 증가분도 고려해야 한다. 이번에는 공간 확보를 위한 삭제·이동을 하지 않았다.

원본 및 archive는 이미 datasets/ Git ignore 적용을 받으며 GitHub 업로드 대상이 아니다. 보고서·probe 코드만 별도로 게시한다. 게시 실패 여부와 재시도 명령은 별도 기록한다.

GitHub 사전 조회(`git ls-remote origin HEAD`)는 `Could not resolve host: github.com`로 실패했다. 원격 게시와 원격 보고서 재확인은 완료되지 않았다. 네트워크 복구 후 재시도: `git -C /home/user/mmdl-work push origin main`. 데이터 다운로드 실패와 보고서 게시 실패는 별개다. 현재 요청의 manifest 사본은 `docs/experiments/E678_raw_download_manifest_20261011.json`에도 보존한다.

실제 `git push origin main`도 같은 DNS 오류로 실패했다. 최초 commit은 Git 작성자 신원 미설정으로 실패했으며, 전역·저장소 환경 설정은 변경하지 않았다.
