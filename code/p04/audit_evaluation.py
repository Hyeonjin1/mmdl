"""Read-only response audit and independent synthetic checks; never starts training/inference."""
import ast
import collections
import importlib.util
import json
import re
import time
from pathlib import Path
from records import ROOT, save, sha, environment, provenance, index, now
import scoring

def main():
    run=ROOT/'runs'/('P04_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_evaluation_audit')
    run.mkdir(); (run/'logs').mkdir()
    source=ROOT/'results/full900_v3/predictions.jsonl'
    cfg={'stage':'evaluation_audit','training_authorized':False,'new_inference':False,
         'purpose':'Audit saved results and synthetic parser cases; no validation-guided rule changes',
         'source':str(source),'model_revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17'}
    save(run/'config_resolved.json',cfg);save(run/'environment.json',environment())
    save(run/'code_provenance.json',provenance())
    save(run/'data_manifest.json',{'source':str(source),'sha256':sha(source),'synthetic_tests':'independently authored; no MMMU GT'})
    save(run/'metrics.json',{'status':'running','started_utc':now()})
    (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/user/MMDL\n.venv-p03/bin/python code/p04/audit_evaluation.py\n')
    (run/'experiment.md').write_text('# Evaluation audit\n\nPurpose: inspect existing artifacts; no additional training or GPU inference.\n')
    spec=importlib.util.spec_from_file_location('old_eval',ROOT/'code/eval_mmmu_v3.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    cases=[('Reasoning (A).\n**Final Answer:** (B)','B'),('### Answer:\n(C)','C'),
           ('Answer: A\nFinal answer: D','D'),('Answer: B or C',None),
           ('Answer: B, actually C',None),('> **Final Answer:** (B)','B'),
           ('- Final answer: (C)','C'),('Reasoning only.\nNo conclusion.',None),
           ('Answer: B. blue','B'),('Answer: (B) and (C)',None)]
    tests=[]
    for response,expected in cases:
        before=old.parse_choice(response,4)[0];candidate=scoring.extract(response,['red','blue','green','black'])['answer']
        tests.append(dict(response=response,expected=expected,existing=before,candidate=candidate,
                          existing_pass=before==expected,candidate_pass=candidate==expected))
    save(run/'synthetic_parser_checks.json',{'tests':tests,'candidate_standard_tests':scoring.selftest(),
         'rules_changed':False,'candidate_frozen_for_inference':False})
    # Image order checked with independent images, including repeated references and options.
    from PIL import Image
    sample={'id':'synthetic','image_1':Image.new('RGB',(28,28),'red'),'image_2':Image.new('RGB',(28,28),'blue')}
    _,images,keys=old.build_content(sample,'<image 2> then <image 1>; option: <image 2>')
    assert keys==['image_2','image_1','image_2'] and images[0].getpixel((0,0))==(0,0,255)
    rows=[json.loads(s) for s in source.read_text().splitlines() if s]
    assert len(rows)==900 and len({r['id'] for r in rows})==900
    mismatch=[r['id'] for r in rows if ['image_'+n for n in re.findall(r'<image\s+(\d+)>',r['prompt'])]!=r['image_keys']]
    def aggregate(rs):
        n=len(rs)
        return {'total':n,'stored_correct':sum(bool(r.get('correct')) for r in rs),
                'stored_parse_failures':sum(bool(r.get('parse_failure')) for r in rs),
                'output_cap_reached':sum(r['generated_tokens']>=r['max_new_tokens'] for r in rs),
                'repetition_stop_count':None,'repetition_note':'No repetition stopping criterion in source code; semantic repetition was not measured'}
    overall=aggregate(rows)
    for key in ['stored_correct','stored_parse_failures','output_cap_reached']:
        overall[key+'_pct']=100*overall[key]/len(rows)
    metrics={'status':'completed','audit_only':True,'optimizer_steps':0,'new_inference':False,
             'overall':overall,'question_types':{k:aggregate([r for r in rows if r['question_type']==k]) for k in sorted({r['question_type'] for r in rows})},
             'subjects':{k:aggregate([r for r in rows if r['subject']==k]) for k in sorted({r['subject'] for r in rows})},
             'token_budgets':dict(collections.Counter(r['max_new_tokens'] for r in rows)),
             'image_reference_order_mismatches':mismatch,'synthetic_image_order_pass':True,
             'existing_parser_synthetic_failures':sum(not t['existing_pass'] for t in tests),
             'candidate_parser_synthetic_failures':sum(not t['candidate_pass'] for t in tests),
             'task_type_labels_available':False,'training_effect':None,'evaluation_change_effect':None,
             'reevaluation_status':'blocked: GPU inaccessible and original MMMU image dataset unavailable',
             'ended_utc':now()}
    save(run/'metrics.json',metrics)
    p03=ROOT/'runs/P03_20261008T075816Z_evaluate'
    pc=json.loads((p03/'config_resolved.json').read_text());pm=json.loads((p03/'metrics.json').read_text())
    save(run/'p03_comparison.json',{'config':pc,'metrics':pm,'raw_responses_present':(p03/'responses.jsonl').exists(),
         'comparison':'same code loop for base/adapter; runtime identity cannot be independently rechecked without raw responses; ChartQA16 only, not MMMU'})
    report=f'''# 평가 절차 감사 — {run.name}

진행 중이던 `P04_20261010T154918Z_prepare_B0`는 부분 가공으로 정상 종료했다. E1/E2 각 1000, ChartQA/TabMWP check 각 100이다. PGPS9K는 0이며 MMMU 이미지 중복 검사는 미실행이다. `ready_for_training=false`이다. 학습이나 후속 GPU 실험을 시작하지 않았다.

## 실제 확인한 기존 결과

저장 응답 900개와 고유 ID 900개를 읽었다. 기존 저장 판정은 {overall['stored_correct']}/900, 추출 실패 {overall['stored_parse_failures']}/900, 출력 상한 도달 {overall['output_cap_reached']}/900이다. 이는 새 B0 평가도, 563/900 혼합 판정의 재현도 아니다. 새 파서로 validation을 재채점하지 않았다.

MMMU 응답에 기록된 상한은 객관식 2048(847개), 주관식 8192(53개)이다. P03 ChartQA16 점검의 64토큰과 혼동하지 않는다. P03의 base 2/16, adapter 8/16은 학습 효과의 증거로 사용하지 않는다. 이 로컬 스냅샷에는 P03 원 응답이 없어 조건 일치 여부는 코드/설정/완료 기록 수준에서만 확인했다.

## 발견한 문제와 확인 범위

- 독립 합성 10개에서 기존 파서 {metrics['existing_parser_synthetic_failures']}개, 미확정 P04 후보 파서 {metrics['candidate_parser_synthetic_failures']}개가 기대 동작과 달랐다. 복수 후보·같은 줄 답 수정·Markdown 형태를 포함한다. 실패 응답과 기대값은 synthetic_parser_checks.json에 보존했다. validation 정답을 이용한 예외 규칙은 추가하지 않았다. 후보 파서는 실평가에 사용할 준비가 되지 않았다.
- 기존 MMMU 코드는 sampling=true, temperature=.7, top_p=.8, top_k=20, presence penalty=1.5, seed=3407이다. P03은 greedy다. 미실행 P04 common.json은 greedy/2048 및 반복 중단 설정을 제안하지만 실행 엔진이 없어 실제 적용된 설정으로 취급할 수 없다.
- 기존 MMMU 코드에 반복 중단 기준은 없다. 반복 중단률을 반복 발생률로 보고하지 않는다. 응답별 생성 token ID가 없어 새 token 반복 중단 정책의 사후 효과는 확인할 수 없다.
- 이미지 참조 순서와 저장 image_keys의 불일치는 {len(mismatch)}개다. 별도 합성 이미지로 2→1→2 순서도 검증했다. 실제 MMMU 원 이미지가 없어 픽셀/내용 대응까지 확인하지는 못했다.
- 기존 MMMU 코드 기본 pixels는 4096/1048576, P03은 200704/1003520이다. 기존 실행의 해상도 인자는 로컬 run_config가 없으므로 기본값이 실제 적용됐다고 단정할 수 없다. 저장 image_grid_thw만으로 모든 processor 조건을 복원할 수 없다.
- 주관식 숫자 정규화는 기존 코드와 P04 후보가 다르다(기존 소수 둘째 자리 반올림 경로, 후보 정확한 분수/단위 유지). 점수 혼합은 불가하다. 길이 상한 도달 자체를 오답으로 바꾸지 않았다.
- E1/E2/E3 완료 adapter나 대응 MMMU 평가는 없다. 따라서 동일 조건의 학습 효과, 목표/비대상 작업 유형 변화는 계산할 수 없다. 작업 유형 라벨 파일도 확인되지 않았다. 과목별 및 객관식/주관식 집계는 metrics.json에 별도로 보존했다.

## 차단 사유와 다음 단계

현재 샌드박스 nvidia-smi는 driver communication failure이고 CUDA 장치에 접근할 수 없다(environment.json). 원 MMMU 이미지와 PGPS9K도 미확보다. CPU 감사만 완료했으며 재추론·추가 학습·자동 후속 실험은 실행하지 않았다. 기존 응답과 실패 기록은 수정하지 않았다.

GPU/원본 데이터가 접근 가능한 환경에서 먼저 합성/dev 파서 실패를 수정하고 prompt/이미지/생성/반복/채점 계약을 고정해야 한다. 그 후 같은 조건으로 base와 이미 완료된 adapter만 평가한다. 신규 학습은 현재 사용자 지시로 보류한다. 평가 절차 변화는 같은 원 응답의 재채점 효과, 학습 효과는 동일 추론/채점 계약의 base–adapter 차이로 별도 보고해야 한다.
'''
    (run/'experiment.md').write_text(report)
    (run/'logs/audit.log').write_text('Completed read-only saved-response audit and synthetic parser/image-order tests. No GPU job launched.\n')
    index();print(run);print(json.dumps({k:v for k,v in metrics.items() if k not in ['subjects','question_types']},ensure_ascii=False))

if __name__=='__main__':main()
