"""Recorded stages and one sequential GPU suite; never regenerate MMMU B0."""
import argparse, collections, datetime, fcntl, hashlib, json, os, shlex, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/p04'),str(ROOT/'code/p03')]
from records import environment,provenance,save,sha,rows,Blocked
from contracts import b0,checkpoint
from rescore_latest import aggregate
CONFIG=ROOT/'configs/restore/E123.json'
STATE=ROOT/'.restore/suite_state.json'

def now():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def record(stage,cfg):
    run=ROOT/'runs'/('P04_'+now()+'_restore_'+stage+'_'+cfg.get('experiment','B0'))
    run.mkdir();(run/'logs').mkdir()
    save(run/'config_resolved.json',cfg);save(run/'code_provenance.json',provenance())
    # Admission must precede CUDA context initialization: otherwise our own process
    # appears as an occupied GPU before the read-only admission check.
    save(run/'environment.json',{'status':'pending admission' if stage in ['train','reload','evaluate','check_base'] else 'pending CPU environment record','python':sys.executable})
    save(run/'data_manifest.json',{'status':'pending'})
    save(run/'metrics.json',{'status':'created','optimizer_steps':0})
    (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])+'\n')
    (run/'experiment.md').write_text('# '+run.name+'\n\nStage: '+stage+'. Original-answer SFT; no teacher/KD. Seed3407. Exact conditions, environment, data hashes, command and code provenance are adjacent JSON/shell files. Results pending.\n')
    return run

def validate(cfg,run):
    import torch
    from learning import encode,processor_for
    from contracts import image_manifest,training_rows
    torch.set_num_threads(1);image_manifest(cfg);p=processor_for(cfg);allrows={}
    manifest=json.loads((Path(cfg['processed_dir'])/'manifest.json').read_text())
    required=['E1','E2','ChartQA_check','TabMWP_check']+(['E3','PGPS9K_check'] if manifest.get('E3_ready') else [])
    for name in required:
        info=manifest['files'][name];assert sha(info['path'])==info['sha256']
        for row in rows(info['path']):allrows[row['id']]=row
    checks=[];samples=[]
    for i,row in enumerate(allrows.values(),1):
        batch,detail=encode(dict(row,original_id=row['id'],chart_name=Path(row['image']).name),p,cfg)
        assert torch.all(batch['labels'][batch['attention_mask']==0]==-100)
        checks.append({k:detail[k] for k in ['original_id','total_tokens','supervised_tokens','image_tokens','image_grid_thw']})
        if sum(s['source']==row['source'] for s in samples)<2:samples.append(dict(detail,source=row['source']))
        if i%50==0:save(run/'progress.json',{'rows_checked':i,'optimizer_steps':0});print('MASK_CHECK',i,len(allrows),flush=True)
    save(run/'token_checks.json',checks);save(run/'mask_samples.json',samples);save(run/'data_manifest.json',manifest)
    return {'rows_checked':len(checks),'assistant_only_mask_verified':True,'truncation':False,
            'max_total_tokens':max(r['total_tokens'] for r in checks),'min_supervised_tokens':min(r['supervised_tokens'] for r in checks),'optimizer_steps':0}

def reload(cfg,run):
    import torch
    from transformers import set_seed
    from peft import PeftModel
    from learning import load_student,processor_for,record_phase
    from PIL import Image
    path,manifest,_=checkpoint(cfg);base=load_student(cfg);model=PeftModel.from_pretrained(base,path,is_trainable=False).eval();p=processor_for(cfg)
    output=[]
    for dataset in cfg['check_datasets']:
        row=rows(Path(cfg['processed_dir'])/(dataset+'_check.jsonl'))[0]
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
        prompt=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        with Image.open(row['image']) as im:image=im.convert('RGB')
        batch=p(text=[prompt],images=[image],return_tensors='pt',truncation=False).to('cuda')
        set_seed(cfg['seed']);record_phase(run,'reload_user_only_generation',0)
        with torch.inference_mode():generated=model.generate(**batch,max_new_tokens=8,do_sample=False)
        tokens=generated[0,batch['input_ids'].shape[1]:];text=p.decode(tokens,skip_special_tokens=True)
        if tokens.numel()==0:raise RuntimeError('Reload generation produced no tokens')
        output.append({'id':row['id'],'generated_tokens':tokens.tolist(),'response':text,'gold_in_input':False})
    save(run/'reload_checks.json',output);save(run/'data_manifest.json',{'checkpoint_manifest':manifest,'independent_process':True})
    return {'reload_verified':True,'independent_process':True,'generation_count':len(output),'verification_only_max_new_tokens':8,'not_scored_as_benchmark':True,'optimizer_steps':0}

def check_base(cfg,run):
    """Only source heldout checks. MMMU IDs cannot be submitted here."""
    import pandas as pd
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import SamplingParams
    from learning import snapshot_path
    from evaluate_vllm import pinned_functions,SOURCE,versions,ESSENTIAL
    from scoring_latest_v2 import extract,judge
    _,protocol,_=b0(cfg)
    current=versions()
    if any((current[k] or '').split('+')[0]!=v for k,v in ESSENTIAL.items()):raise Blocked('Pinned evaluation package mismatch:'+str(current))
    for name,h in protocol['checkpoint_contract']['guard_code_sha256'].items():
        if sha(SOURCE/name)!=h:raise Blocked('Historical repetition policy code differs')
    sys.path.insert(0,str(SOURCE));from streaming_guard import StreamingGuardEngine
    _,prepare=pinned_functions({'pd':pd,'process_vision_info':process_vision_info})
    processor=AutoProcessor.from_pretrained(snapshot_path(cfg),local_files_only=True)
    args=dict(protocol['engine_args']);args['model']=snapshot_path(cfg);engine=StreamingGuardEngine(args)
    s=protocol['settings'];params=SamplingParams(**{k:s[k] for k in ['temperature','top_p','top_k','repetition_penalty','presence_penalty']},max_tokens=8192,seed=3407,stop_token_ids=[])
    results=[]
    try:
        for dataset in cfg['check_datasets']:
            data=rows(Path(cfg['processed_dir'])/(dataset+'_check.jsonl'))
            if cfg.get('smoke'):data=data[:1]
            for row in data:
                if row['id'].startswith('validation_') or row['source']=='MMMU':raise Blocked('MMMU original inference forbidden')
                messages=[{'role':'user','content':[{'type':'image','image':row['image'],'min_pixels':1003520,'max_pixels':4014080},
                    {'type':'text','text':'Question: '+row['question']+protocol['prompt_protocol']['cot_suffix']}]}]
                prepared=prepare(messages,processor)
                count=processor(text=[prepared['prompt']],images=prepared['multi_modal_data'].get('image'),return_tensors='pt',padding=False,truncation=False)['input_ids'].shape[1]
                if count+8192>49152:raise Blocked('Heldout context exceeds historical budget')
                out=engine.generate([prepared],sampling_params=params,use_tqdm=False)[0].outputs[0]
                options={chr(65+i):v for i,v in enumerate(row.get('options',[]))};e=extract(out.text.split('</think>')[-1].strip(),options)
                aborted=out.finish_reason=='repetition_abort';item={'id':row['id'],'dataset':dataset,'subject':dataset,'question_type':'multiple-choice' if options else 'open',
                    'gold':row['answer'],'response_raw':out.text,'extracted_answer':None if aborted else e['answer'],'correct':not aborted and judge(e,row['answer'],options),
                    'parse_failure':not aborted and e['answer'] is None,'missing_final_answer':e['failure']=='missing_final_answer','explicit_incorrect':not aborted and e['answer'] is not None and not judge(e,row['answer'],options),
                    'finish_reason':out.finish_reason,'output_cap_reached':len(out.token_ids)>=8192,'output_tokens':len(out.token_ids),'input_tokens':count}
                results.append(item)
                with (run/'responses.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                save(run/'progress.json',{'completed_questions':len(results),'optimizer_steps':0});print('BASE_CHECK',dataset,row['id'],out.finish_reason,flush=True)
    finally:engine.close()
    save(run/'data_manifest.json',{'check_manifest_sha256':sha(Path(cfg['processed_dir'])/'manifest.json'),'MMMU_inference_count':0,'B0_protocol_sha256':sha(Path(cfg['b0_run'])/'evaluation_protocol.json')})
    return {'optimizer_steps':0,'original_MMMU_inference_count':0,'original_heldout_inference_count':len(results),'smoke':cfg.get('smoke',False),'check_scores':{d:aggregate([r for r in results if r['dataset']==d]) for d in cfg['check_datasets']}}

def stage(args):
    cfg=json.loads(CONFIG.read_text());cfg.update(stage=args.stage,experiment=args.experiment,smoke=args.smoke,adapter_path=args.adapter)
    run=record(args.stage,cfg);lock=None;start=time.perf_counter()
    try:
        if args.stage in ['train','reload','evaluate','check_base']:
            (ROOT/'.locks').mkdir(exist_ok=True);lock=(ROOT/'.locks/restore-gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from gpu_admission import select_gpu
            admission=select_gpu();save(run/'gpu_selection.json',admission)
            if admission.get('error'):raise Blocked(admission['error'])
            # vLLM0.11.2 interprets CUDA_VISIBLE_DEVICES as integer indices.
            # Only translate the selected UUID on this verified single-GPU host.
            devices=[line for line in admission['devices']['stdout'].splitlines() if line.strip()]
            if len(devices)!=1:raise Blocked('Numeric vLLM device mapping requires reviewed single-GPU host')
            os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES']=str(admission['selected']['physical_index'])
            admission['actual_cuda_visible_devices']=os.environ['CUDA_VISIBLE_DEVICES']
            admission['numeric_mapping_reason']='vLLM0.11.2 cannot parse GPU UUID; single physical GPU verified'
            save(run/'gpu_selection.json',admission)
            save(run/'environment.json',environment())
            if not (Path(cfg['model_path'])/'model_manifest.json').exists():raise Blocked('Complete pinned weights not verified')
        else:save(run/'environment.json',environment())
        import engine
        with (run/'logs/stage.log').open('x',buffering=1) as log:
            import contextlib
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                result={'validate':validate,'train':engine.train,'reload':reload,'evaluate':engine.evaluate,'check_base':check_base}[args.stage](cfg,run)
        result.update(status='completed')
    except BaseException as e:
        result={'status':'blocked' if isinstance(e,(Blocked,BlockingIOError)) else 'oom' if 'out of memory' in str(e).lower() else 'error','error':repr(e)}
        save(run/'logs/error.json',{'error':repr(e),'traceback':traceback.format_exc(),'last_phase':json.loads((run/'execution_phase.json').read_text()) if (run/'execution_phase.json').exists() else None})
    finally:
        if lock:lock.close()
    for name in ['progress.json','execution_phase.json']:
        if (run/name).exists():result={**json.loads((run/name).read_text()),**result}
    result.update(wall_seconds=time.perf_counter()-start,ended_utc=now());save(run/'metrics.json',result)
    with (run/'experiment.md').open('a') as f:f.write('\n## Result\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\nFailures preserve completed work and prevent dependent evaluation. No unrecorded precision, token or image-setting changes.\n')
    print(run,result['status'],flush=True)
    return 0 if result['status']=='completed' else 1

def suite():
    cfg=json.loads(CONFIG.read_text());(ROOT/'.locks').mkdir(exist_ok=True)
    with (ROOT/'.locks/restore-suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code={str(p.relative_to(ROOT)):sha(p) for folder in ['code/restore','code/p03','code/p04'] for p in (ROOT/folder).glob('*.py')}
        signature=hashlib.sha256(json.dumps({'cfg':cfg,'code':code},sort_keys=True).encode()).hexdigest()
        state=json.loads(STATE.read_text()) if STATE.exists() else {'signature':signature,'completed':{},'history':[]}
        if state['signature']!=signature:raise Blocked('Suite source/config changed; preserve receipts and deliberately create a new suite')
        if state.get('failed'):raise Blocked('Previous failure requires review; do not automatically retry training')
        state.update(pid=os.getpid(),status='running');save(STATE,state)
        def execute(key,name,experiment='B0',adapter=None,smoke=False):
            if key in state['completed']:
                prior=Path(state['completed'][key]);assert json.loads((prior/'metrics.json').read_text())['status']=='completed';return prior
            before=set((ROOT/'runs').iterdir());python=cfg['eval_python'] if name in ['evaluate','check_base'] else cfg['train_python']
            cmd=[python,str(Path(__file__).resolve()),'--stage',name,'--experiment',experiment]
            if adapter:cmd+=['--adapter',str(adapter)]
            if smoke:cmd+=['--smoke']
            state.update(active_stage=key,active_command=cmd);save(STATE,state)
            print('START',key,shlex.join(cmd),flush=True)
            result=subprocess.run(cmd,cwd=ROOT)
            created=set((ROOT/'runs').iterdir())-before
            if len(created)!=1:raise Blocked('Stage receipt not uniquely found')
            run=created.pop();state['history'].append({'key':key,'run':str(run),'returncode':result.returncode})
            if result.returncode:
                state.update(status='failed',failed=key,failed_run=str(run));save(STATE,state);raise Blocked('Stage failed; dependent steps stopped:'+key)
            state['completed'][key]=str(run);save(STATE,state);return run
        # Reuse a completed full CPU audit only when data and encoding settings agree.
        if 'validate' not in state['completed']:
            manifest=Path(cfg['processed_dir'])/'manifest.json'
            for prior in sorted((ROOT/'runs').glob('*restore_validate_B0'),reverse=True):
                if not (prior/'metrics.json').exists():continue
                metrics=json.loads((prior/'metrics.json').read_text());settings=json.loads((prior/'config_resolved.json').read_text())
                same=all(settings[k]==cfg[k] for k in ['model_revision','min_pixels','max_pixels','max_sequence_length','processed_dir','seed'])
                if metrics.get('status')=='completed' and same and sha(prior/'data_manifest.json')==sha(manifest):
                    state['completed']['validate']=str(prior);save(STATE,state);break
        execute('validate','validate')
        execute('backend_check','check_base',smoke=True)
        smoke=execute('smoke_train','train','smoke',smoke=True)
        execute('smoke_reload','reload','smoke',smoke/'adapter',True)
        for experiment in ['E1','E2','E3']:
            manifest=json.loads((Path(cfg['processed_dir'])/'manifest.json').read_text())
            if experiment=='E3' and not manifest.get('E3_ready'):
                state['E3_status']='blocked: PGPS preparation incomplete';save(STATE,state);continue
            train=execute(experiment+'_train','train',experiment)
            execute(experiment+'_reload','reload',experiment,train/'adapter')
            if experiment=='E1':
                execute('smoke_eval','evaluate','smoke',smoke/'adapter',True)
            execute(experiment+'_evaluate','evaluate',experiment,train/'adapter')
        # Source-check base inference is independent of all training and may wait.
        execute('heldout_base','check_base')
        state.update(status='completed' if 'E3_evaluate' in state['completed'] else 'E12_completed_E3_blocked',active_stage=None);save(STATE,state)
    return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--stage',choices=['validate','train','reload','evaluate','check_base']);p.add_argument('--experiment',default='B0',choices=['B0','smoke','E1','E2','E3']);p.add_argument('--adapter');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.status:print(STATE.read_text() if STATE.exists() else 'Suite not started');raise SystemExit(0)
    if a.suite:raise SystemExit(suite())
    if not a.stage:p.error('--stage or --suite required')
    raise SystemExit(stage(a))
