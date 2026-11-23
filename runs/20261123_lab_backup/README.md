# MMDL 실습실 실험 백업

- `lookback-complete.tar.gz`에 Lookback 80쌍의 원문 예측·개입·비교와 smoke 결과가 있다.
- `lookback-reproduction.tar.gz`에 실행 및 검사 코드, 팀 가드 어댑터, 실행 설정, 원본 이미지 매니페스트와 추적 해시가 있다. 팀 checkout `1c4d7b32ab254d8444b401b06750dfb6e2bfca9e` 위에 코드를 배치하고 매니페스트 주소에 같은 모델 및 이미지 캐시를 준비한다. 실행 명령은 결과 폴더의 `experiment.md`에 있다.
- `archives/MANIFEST.json`에 팀 B0·parser v2, 실습실 B0·Judge, 두 최근 파일럿의 전체 백업 목록과 SHA-256이 있다. NoWait는 캡처 시점의 중간 결과다.

분할 압축 파일은 같은 폴더에서 `cat team-results.tar.gz.part-* > team-results.tar.gz`처럼 이름 순으로 합쳐 검증한 뒤 `tar -xzf team-results.tar.gz`로 풀 수 있다. 다른 분할 압축에도 같은 절차를 적용한다.
