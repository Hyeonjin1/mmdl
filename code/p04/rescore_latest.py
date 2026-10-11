"""CPU-only B0 from exact recent vLLM saved responses; never invokes a model."""
import argparse,collections,csv,hashlib,json,re,traceback
from pathlib import Path
from records import ROOT,save,sha,index,now
import scoring_latest_v2 as scoring

SNAP=ROOT/'review_snapshots/0c0b415f6f93923a4249d3da32c5cbb447a518ea'

def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def aggregate(rows):
    n=len(rows);counts=dict(total=n,correct=sum(r['correct'] for r in rows),
      parse_failures=sum(r['parse_failure'] for r in rows),missing_final_answers=sum(r['missing_final_answer'] for r in rows),
      explicit_incorrect=sum(r['explicit_incorrect'] for r in rows),
      length_finish=sum(r['finish_reason']=='length' for r in rows),output_cap_reached=sum(r['output_cap_reached'] for r in rows),
      repetition_aborts=sum(r['finish_reason']=='repetition_abort' for r in rows),execution_errors=0)
    for k in ['correct','parse_failures','missing_final_answers','length_finish','output_cap_reached','repetition_aborts']:
        counts[k+'_pct']=100*counts[k]/n if n else None
    return counts

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    if (run/'rescored.jsonl').exists():raise RuntimeError('Existing B0 preserved; do not duplicate rescoring')
    try:
        tests=scoring.selftest();save(run/'parser_synthetic_checks.json',tests)
        predpath=SNAP/'results/validation900_cot8192_guard_full/predictions.jsonl'
        mixedpath=SNAP/'results/validation900_rules_v1/judgments.jsonl'
        srcconfig=SNAP/'results/validation900_cot8192_guard_full/run_config.json'
        rc=json.loads(srcconfig.read_text());rows=read(predpath);mixed=read(mixedpath);mixed_by={r['question_id']:r for r in mixed}
        assert len(rows)==900 and len({r['question_id'] for r in rows})==900
        assert set(mixed_by)=={r['question_id'] for r in rows}
        assert rc['selected_ids']==[r['question_id'] for r in rows]
        assert collections.Counter(r['annotation']['category'] for r in rows)==collections.Counter({s:30 for s in {r['annotation']['category'] for r in rows}})
        assert len({r['annotation']['category'] for r in rows})==30
        assert sum(r['correct'] is True for r in mixed)==563
        assert {r['checkpoint_contract_hash'] for r in rows}=={rc['checkpoint_contract_hash']}
        assert {r['max_new_tokens'] for r in rows}=={8192}
        assert rc['engine_args']['model'].endswith('ebb281ec70b05090aa6165b016eac8ec08e71b17')
        protocol={'version':'P04_latest_vllm_v1','model_id':'Qwen/Qwen3-VL-4B-Instruct',
          'model_revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17',
          'source_run_config_sha256':sha(srcconfig),'source_predictions_sha256':sha(predpath),
          'source_mixed_judgments_sha256':sha(mixedpath),'dataset_manifest_sha256':rc['dataset_sha256'],
          'settings':rc['settings'],'engine_args':rc['engine_args'],
          'prompt_protocol':rc['prompt_protocol'],'repetition_policy':rc['repetition_policy'],
          'upstream_commit':rc['upstream_commit'],'environment':rc['environment'],
          'checkpoint_contract':rc['checkpoint_contract'],
          'scoring':{'version':scoring.VERSION,'sha256':sha(ROOT/'code/p04/scoring_latest_v2.py'),
            'normalization_dependency_sha256':sha(ROOT/'code/p04/scoring_v2.py'),
            'repetition_abort':'policy incorrect, extraction not attempted; raw answer availability separate',
            'length_limit':'retain any extractable answer, never automatically incorrect',
            'open':'exact Fraction and unit-preserving text/alias normalization; no tolerance or unit conversion',
            'mc':'GT-blind explicit-final/terminal answer, letter+body requires exact option agreement',
            'extraction_GT_access':False},
          'image_order':'original image_N ascending, all images before text; preserve saved message content',
          'original_model_inference_allowed':False,'adapter_backend_required':'vllm==0.11.2',
          'is_official_MMMU_score':False,'validation_is_development_comparison':True}
        save(run/'evaluation_protocol.json',protocol)
        # First extract all answers without gold, old correctness or judge verdicts.
        extractions=[];format_checks=[]
        for r in rows:
            ann=r['annotation'];options={k:ann[k] for k in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if ann.get(k) is not None}
            text=r['result']['gen_raw'].split('</think>')[-1].strip()
            e=scoring.extract(text,options)
            extractions.append((r['question_id'],e,text,options))
            image_items=[c for m in r['messages'] for c in m['content'] if c['type']=='image']
            assert [c['image'] for c in image_items]==ann['image_path']
            assert all((c['min_pixels'],c['max_pixels'])==(1003520,4014080) for c in image_items)
            image_numbers=[int(re.search(r'_(\d+)\.png$',c['image'])[1]) for c in image_items]
            assert image_numbers==list(range(1,len(image_items)+1))
            refs=[int(n) for n in re.findall(r'<image\s+(\d+)>',ann['question']+'\n'+'\n'.join(options.values()))]
            assert set(refs).issubset(image_numbers)
            # No gold appears as a separately appended answer message/input annotation.
            assert all(m['role']=='user' for m in r['messages'])
            format_checks.append({'id':r['question_id'],'image_count':len(image_items),'image_numbers':image_numbers,'reference_occurrences':refs})
        save(run/'image_reference_checks.json',format_checks)
        out=[];disagreements=[]
        for r,(_,raw_e,text,options) in zip(rows,extractions):
            ann=r['annotation'];aborted=r['finish_reason']=='repetition_abort';e=raw_e if not aborted else {'answer':None,'method':'repetition_abort_policy','failure':None}
            # Gold is read only after extraction is finished.
            correct=not aborted and scoring.judge(e,ann['answer'],options)
            old=mixed_by[r['question_id']]
            assert old['gold']==ann['answer']
            item={'experiment_id':run.name,'id':r['question_id'],'subject':ann['category'],'question_type':ann['question_type'],
              'gold':ann['answer'],'response_raw':r['result']['gen_raw'],'extracted_answer':e['answer'],
              'extraction_method':e['method'],'extraction_failure':e['failure'],'correct':correct,
              'parse_failure':not aborted and e['answer'] is None,'missing_final_answer':raw_e['answer'] is None and raw_e['failure']=='missing_final_answer',
              'explicit_incorrect':not aborted and e['answer'] is not None and not correct,
              'input_tokens':r['input_tokens'],'output_tokens':r['generated_tokens'],'max_new_tokens':8192,
              'output_cap_reached':r['generated_tokens']>=8192,'finish_reason':r['finish_reason'],
              'elapsed_seconds':r['elapsed_seconds'],'mixed_correct':old['correct'] is True,
              'raw_answer_if_aborted':raw_e['answer'] if aborted else None,'retry':False,'execution_error':None}
            out.append(item)
            old_prediction=old.get('prediction')
            if old_prediction is None and isinstance(old.get('raw_judge_response'),str):
                try:
                    verdict=json.loads(old['raw_judge_response']);old_prediction=verdict.get('prediction',verdict.get('answer'))
                except (ValueError,AttributeError):pass
            if old_prediction!=e['answer'] or (old['correct'] is True) != correct:
                # Evidence file excludes GT to support format review without correctness tuning.
                disagreements.append({'id':r['question_id'],'question_type':ann['question_type'],
                  'response_tail':text[-1600:],'options':options,'previous_extracted':old_prediction,'new_extraction':e,
                  'finish_reason':r['finish_reason'],'review_status':'unreviewed_format_case; no GT supplied'})
        (run/'rescored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
        (run/'extraction_disagreements_no_gt.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in disagreements))
        metrics={'status':'completed','kind':'CPU_saved_recent_vllm_rescore_not_new_inference','optimizer_steps':0,
          'source_correct_mixed':563,'source_total':900,'source_accuracy_pct_mixed':100*563/900,
          'overall':aggregate(out),'by_subject':{s:aggregate([r for r in out if r['subject']==s]) for s in sorted({r['subject'] for r in out})},
          'by_question_type':{s:aggregate([r for r in out if r['question_type']==s]) for s in ['multiple-choice','open']},
          'changed_correctness_from_mixed':sum(r['correct']!=r['mixed_correct'] for r in out),
          'mixed_correct_to_new_incorrect':sum(r['mixed_correct'] and not r['correct'] for r in out),
          'mixed_incorrect_to_new_correct':sum(not r['mixed_correct'] and r['correct'] for r in out),
          'raw_extraction_disagreement_candidates':len(disagreements),'task_type_labels':'not available; subject is not substituted',
          'training_effect':None,'evaluation_protocol_sha256':sha(run/'evaluation_protocol.json'),'ended_utc':now()}
        metrics['rescoring_delta_pp']=metrics['overall']['correct_pct']-metrics['source_accuracy_pct_mixed']
        save(run/'metrics.json',metrics)
        save(run/'data_manifest.json',{'source_repository':'eses0070/MMDL','commit':'0c0b415f6f93923a4249d3da32c5cbb447a518ea',
          'files':json.loads((run/'source_hashes.json').read_text()),'id_count':900,'image_files_acquired':False,
          'recent_annotation_path':str(predpath),'expected_original_image_export_manifest_sha256':rc['dataset_sha256']})
        fields=['id','subject','question_type','extracted_answer','correct','extraction_method','finish_reason','input_tokens','output_tokens']
        (run/'public_predictions.jsonl').write_text(''.join(json.dumps({k:r[k] for k in fields},ensure_ascii=False)+'\n' for r in out))
        with (run/'subject_metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['subject',*next(iter(metrics['by_subject'].values())).keys()]);w.writeheader();w.writerows({'subject':s,**v} for s,v in metrics['by_subject'].items())
        (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/user/MMDL\n# Create a NEW recorded run before rescoring; existing rescored.jsonl is never overwritten.\n.venv-p03/bin/python code/p04/rescore_latest.py --run "$1"\n')
        o=metrics['overall']
        (run/'experiment.md').write_text(f'''# Recent vLLM B0 — {run.name}

목적: 최근900 저장 응답을 CPU에서 일관되게 채점. 원본 모델 재추론 없음.

## 실제 실행과 결과

고정 GitHub commit의 원응답 900개와 혼합 판정 900개를 Git blob/SHA-256으로 검증했다(source_hashes.json). 초기493/900 기록을 사용하지 않았다. 모든 답을 GT 없이 추출한 뒤 GT와 비교했다. 독립 합성21개 통과; 실제 형식 불일치 후보는 extraction_disagreements_no_gt.jsonl에 GT 없이 남겼다.

기존 혼합 점수: 563/900 = 62.56%. 새 일관 규칙 B0: {o['correct']}/900 = {o['correct_pct']:.2f}%. 변화 {metrics['rescoring_delta_pp']:+.2f}%p는 동일 응답의 채점 변경 효과이며 학습 효과가 아니다. 혼합 정답→규칙 오답 {metrics['mixed_correct_to_new_incorrect']}개, 반대 {metrics['mixed_incorrect_to_new_correct']}개.

추출 실패 {o['parse_failures']}/900 ({o['parse_failures_pct']:.2f}%), finish length {o['length_finish']}/900 ({o['length_finish_pct']:.2f}%), 실제8192토큰 도달 {o['output_cap_reached']}/900 ({o['output_cap_reached_pct']:.2f}%), 반복 중단 {o['repetition_aborts']}/900 ({o['repetition_aborts_pct']:.2f}%). 반복 중단은 별도 정책 오답이며 추출 실패에 중복 집계하지 않았다. length는 추출 가능한 답을 그대로 채점했다. 원문은 로컬 rescored.jsonl에 보존했다.

## 고정 추론 및 채점 계약

evaluation_protocol.json은 최근 실제 run_config에서 추출했다. vLLM0.11.2, 동일 pinned model, 이미지 min1003520/max4014080, image_N 순서로 모든 이미지를 텍스트 앞에 배치, CoT, max output8192/context49152, sampling .7/.8/top_k20/presence1.5, seed3407, 동일 streaming 반복 정책이다. P03의64토큰과 초기MC2048토큰을 사용하지 않는다.

## 한계와 다음 단계

공식 점수가 아닌 개발 비교셋 규칙 점수다. 엄격한 옵션 본문/숫자/단위/정답 문자열 일치 때문에 의미상 동등한 답도 놓칠 수 있다. reasoning을 검증하지 않았고 tolerance를 느슨하게 만들지 않았다. 작업 유형 라벨이 없어 유형별/비대상 성능 변화는 아직 산출할 수 없다. 과목/문제 형식별 집계는 metrics.json/subject_metrics.csv이다.

B0 재채점은 완료했으나 Codex GPU 접근과 MMMU 원본 이미지가 미확인이다. E1/E2 완료 학습은 발견되지 않았으며 자동 시작하지 않았다. PGPS9K는 E1/E2/B0의 필수 조건으로 취급하지 않는다. 원본 이미지 검증 및 host GPU probe를 통과한 후 독립 원본 초기화 학습과 adapter만의 동일 vLLM 평가를 실행한다. 원본 모델은 재추론하지 않는다.
''')
        (run/'logs/rescore.log').write_text(json.dumps({k:v for k,v in metrics.items() if k not in ['by_subject','by_question_type']},indent=2)+'\n')
        index();print(run);print(json.dumps(metrics['overall']));print('rescoring_delta_pp',metrics['rescoring_delta_pp'])
    except BaseException as e:
        save(run/'metrics.json',{'status':'error','error':repr(e),'optimizer_steps':0});(run/'logs/error.log').write_text(traceback.format_exc());index();raise

if __name__=='__main__':main()
