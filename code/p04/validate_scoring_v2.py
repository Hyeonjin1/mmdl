"""Synthetic-only development validation; no model, dataset answers or training."""
import json,time
from records import ROOT,save,sha,environment,provenance,index
import scoring_v2 as scorer

def main():
    run=ROOT/'runs'/('P04_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_parser_dev')
    run.mkdir();(run/'logs').mkdir()
    save(run/'config_resolved.json',{'stage':'parser_dev','version':scorer.VERSION,'inference_enabled':False})
    save(run/'environment.json',environment());save(run/'code_provenance.json',provenance())
    save(run/'data_manifest.json',{'source':'independent synthetic examples only','evaluation_GT_used':False})
    save(run/'metrics.json',{'status':'running'})
    (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/user/MMDL\n.venv-p03/bin/python code/p04/validate_scoring_v2.py\n')
    (run/'experiment.md').write_text('# Independent parser development validation\n\nNo training or model inference.\n')
    prior=ROOT/'runs/P04_20261010T155357Z_evaluation_audit/synthetic_parser_checks.json'
    cases=json.loads(prior.read_text())['tests']
    for text in ['Answer: B actually C','Answer: B instead C']:
        cases.append({'response':text,'expected':None})
    outcomes=[]
    for c in cases:
        result=scorer.extract(c['response'],['red','blue','green','black'])
        outcomes.append({'response':c['response'],'expected':c['expected'],'actual':result,'passed':result['answer']==c['expected']})
    standard=scorer.selftest()
    save(run/'synthetic_cases.json',outcomes)
    failures=sum(not r['passed'] for r in outcomes)
    metrics={'status':'completed' if not failures else 'error','synthetic_cases':len(outcomes)+standard['synthetic_cases'],
             'failures':failures,'optimizer_steps':0,'new_inference':False,'parser_sha256':sha(ROOT/'code/p04/scoring_v2.py'),
             'validation_rescoring':False,'production_protocol_frozen':False}
    save(run/'metrics.json',metrics)
    (run/'logs/tests.log').write_text(json.dumps(metrics,indent=2)+'\n')
    (run/'experiment.md').write_text(f'''# 파서 개발 검증 — {run.name}

목적: 기존 결과를 보존하고 독립 합성 예제로 Markdown 최종 답 및 모호한 같은 줄 답 수정 처리를 검증한다.

실행: code/p04/scoring_v2.py 별도 버전. 사용자 입력·모델 호출·MMMU 정답을 사용하지 않았다. 기존 scoring.py 및 기존 응답/판정은 변경하지 않았다. 인용/목록 Markdown 접두사를 허용하고, 같은 줄 actually/instead/rather 답 수정은 추출 실패로 보수적으로 처리한다. 여러 명시적 답 줄은 마지막 답을 선택한다. 숫자/단위 허용 오차는 바꾸지 않았다.

결과: 독립 합성 {metrics['synthetic_cases']}개, 실패 {failures}개. 개별 사례는 synthetic_cases.json, 코드 hash는 metrics.json에 기록했다.

한계: 합성 검증만 완료. 이미지 처리/생성/중단/채점 통합 계약은 아직 확정되지 않았다. 재채점 효과나 학습 효과를 측정하지 않았다. GPU 접근 및 MMMU 원본 이미지 부재로 base/adapter 재평가는 미실행이다.

다음: 별도 dev 이미지 문항에서 통합 평가 절차를 검증한 뒤 base 및 완료 adapter에 동일 조건을 적용한다. 추가 학습 및 자동 후속 실험은 시작하지 않는다.
''')
    index();print(run);print(json.dumps(metrics));return bool(failures)

if __name__=='__main__':raise SystemExit(main())
