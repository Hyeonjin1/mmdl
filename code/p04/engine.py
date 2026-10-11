"""P04 E1/E2 training, reload and comparison; every GPU stage runs via launch.py."""
import collections,gc,json,random,time
from pathlib import Path
from records import ROOT,save,sha,rows,Blocked
from contracts import b0,image_manifest,training_rows,checkpoint,digest

def validate(cfg,run):
    from learning import encode,processor_for
    import torch
    torch.set_num_threads(1)
    data,_=training_rows(cfg,run,cfg.get('smoke',False));p=processor_for(cfg);summary=[];samples=[]
    for row in data:
        row=dict(row,original_id=row['id'],chart_name=Path(row['image']).name)
        inputs,detail=encode(row,p,cfg)
        summary.append({k:detail[k] for k in ['original_id','total_tokens','supervised_tokens','image_tokens','image_grid_thw']})
        if len(samples)<2 or (row['source']=='TabMWP' and not any(s.get('source')=='TabMWP' for s in samples)):
            samples.append(dict(detail,source=row['source']))
        del inputs
    save(run/'mask_samples.json',samples);save(run/'validation_checks.json',{'rows_checked':len(data),'checks':summary,
        'no_truncation':True,'padding':False,'assistant_answer_and_im_end_supervised':True,'user_and_image_loss_masked':True})
    return {'rows_checked':len(data),'max_total_tokens':max(r['total_tokens'] for r in summary),
      'min_supervised_tokens':min(r['supervised_tokens'] for r in summary),'mask_check':'passed'}

def train(cfg,run):
    import torch
    from transformers import set_seed
    from common import require_gpu
    from learning import processor_for,encode,load_student,attach_lora,record_phase
    smoke=cfg.get('smoke',False);data,_=training_rows(cfg,run,smoke)
    expected_steps=cfg['smoke_optimizer_steps'] if smoke else cfg['optimizer_steps']
    warmup=cfg['smoke_warmup_steps'] if smoke else cfg['warmup_steps']
    if cfg['micro_batch']!=1 or cfg['gradient_accumulation']!=4 or len(data)!=4*expected_steps:
        raise Blocked('Exactly one epoch, micro1/accum4 and actual update contract must agree')
    if not smoke and (len(data)!=1000 or expected_steps!=250):raise Blocked('Main experiment requires1000 unique QA/250 updates')
    if cfg.get('adapter_path'):raise Blocked('Training must initialize original model without any adapter')
    gpu=require_gpu(cfg);set_seed(cfg['seed']);validate(cfg,run);p=processor_for(cfg)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();record_phase(run,'model_loading',0)
    model=load_student(cfg);record_phase(run,'lora_attachment',0);model,details=attach_lora(model,cfg,run)
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads();model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    order=list(range(len(data)));random.Random(cfg['seed']).shuffle(order);exposures=collections.Counter();steps=[]
    input_tokens=supervised_tokens=0;optimizer.zero_grad(set_to_none=True)
    with (run/'logs/train_steps.jsonl').open('x',buffering=1) as log:
        for step in range(1,expected_steps+1):
            lr=cfg['learning_rate']*min(step/warmup,1.)
            for group in optimizer.param_groups:group['lr']=lr
            losses=[];ids=[]
            for micro,idx in enumerate(order[(step-1)*4:step*4],1):
                source=data[idx];row=dict(source,original_id=source['id'],chart_name=Path(source['image']).name)
                batch,detail=encode(row,p,cfg);record_phase(run,'batch_transfer',step-1,micro_batch=micro)
                batch={k:v.to('cuda:0') for k,v in batch.items()};record_phase(run,'forward',step-1,micro_batch=micro)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
                if not torch.isfinite(loss):raise FloatingPointError('NaN/Inf training loss')
                record_phase(run,'backward',step-1,micro_batch=micro);(loss/4).backward()
                losses.append(float(loss.detach()));ids.append(source['id']);exposures[source['id']]+=1
                input_tokens+=detail['total_tokens'];supervised_tokens+=detail['supervised_tokens']
                save(run/'progress.json',{'optimizer_steps':step-1,'question_exposures':sum(exposures.values()),
                  'unique_questions_exposed':len(exposures),'completed_micro_batches_in_current_update':micro,
                  'input_tokens':input_tokens,'supervised_tokens':supervised_tokens})
                del batch,loss
            record_phase(run,'gradient_clipping',step-1)
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True)
            if not any(p.grad is not None for p in params):raise RuntimeError('No trainable LoRA gradient')
            record_phase(run,'optimizer_step',step-1);optimizer.step();record_phase(run,'optimizer_update_completed',step)
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
            metric={'optimizer_step':step,'micro_batches':4,'mean_micro_loss':sum(losses)/4,'micro_losses':losses,
              'learning_rate':lr,'grad_norm':float(norm),'question_ids':ids,'question_exposures':sum(exposures.values()),
              'unique_questions_exposed':len(exposures),'input_tokens_cumulative':input_tokens,
              'supervised_tokens_cumulative':supervised_tokens,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.perf_counter()-started}
            steps.append(metric);log.write(json.dumps(metric)+'\n');print('UPDATE',json.dumps(metric),flush=True)
            save(run/'progress.json',{'optimizer_steps':step,'question_exposures':sum(exposures.values()),'unique_questions_exposed':len(exposures),
              'input_tokens':input_tokens,'supervised_tokens':supervised_tokens,'peak_allocated_bytes':metric['peak_allocated_bytes'],
              'peak_reserved_bytes':metric['peak_reserved_bytes']})
    if set(exposures.values())!={1} or len(exposures)!=len(data):raise RuntimeError('One epoch exposure identity check failed')
    if not any(bool(torch.count_nonzero(p).item()) for name,p in model.named_parameters() if 'lora_B' in name):raise RuntimeError('No nonzero trained LoRA B')
    record_phase(run,'checkpoint_save',expected_steps);adapter=run/'adapter'
    model.save_pretrained(adapter,safe_serialization=True);p.save_pretrained(adapter)
    hashes={f.name:sha(f) for f in adapter.iterdir() if f.is_file()}
    save(run/'checkpoint_manifest.json',{'base_model_id':cfg['model_id'],'base_revision':cfg['model_revision'],
      'files_sha256':hashes,'data_manifest_sha256':sha(run/'data_manifest.json'),'optimizer_steps':expected_steps,
      'question_exposures':len(data),'experiment':cfg['experiment'],'smoke':smoke,
      'B0_protocol_sha256':sha(Path(cfg['b0_run'])/'evaluation_protocol.json')})
    save(run/'exposures.json',{'per_question':dict(exposures),'total':sum(exposures.values()),'unique':len(exposures),'epochs_equivalent':1.})
    return {'optimizer_steps':expected_steps,'adapter_path':str(adapter),'adapter_saved':True,'reload_verified':False,
      'question_exposures':len(data),'unique_questions_exposed':len(data),'epochs_equivalent':1.,'input_tokens':input_tokens,
      'supervised_tokens':supervised_tokens,'first_loss':steps[0]['mean_micro_loss'],'last_loss':steps[-1]['mean_micro_loss'],
      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
      'training_seconds':time.perf_counter()-started,'trainable_parameter_count':details['trainable_parameter_count'],
      'actual_gpu':gpu,'smoke':smoke,'actual_warmup_updates':warmup,'original_initialization':True}

def reload(cfg,run):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from common import require_gpu
    from learning import load_student,processor_for,encode,record_phase
    require_gpu(cfg);path,manifest,metrics=checkpoint(cfg);_,protocol,_=b0(cfg)
    tensors=load_file(str(path/'adapter_model.safetensors'),device='cpu')
    if not tensors or any(not torch.isfinite(t).all() for t in tensors.values()):raise RuntimeError('Invalid adapter tensors')
    if not all('language_model' in n and 'lora_' in n and not any(k in n for k in ['visual','merger']) for n in tensors):raise RuntimeError('Adapter modifies non-language module')
    record_phase(run,'reload_model_loading',0);base=load_student(cfg);model=PeftModel.from_pretrained(base,str(path),is_trainable=False);model.eval()
    p=processor_for(cfg);checks=[]
    for name in ['ChartQA_check','TabMWP_check']:
        row=rows(Path(cfg['processed_dir'])/(name+'.jsonl'))[0]
        row=dict(row,original_id=row['id'],chart_name=Path(row['image']).name)
        batch,detail=encode(row,p,cfg);batch.pop('labels');batch={k:v.to('cuda:0') for k,v in batch.items()}
        record_phase(run,'reload_image_forward',0)
        with torch.inference_mode():output=model(**batch,use_cache=False)
        if not torch.isfinite(output.logits[:,-1]).all():raise FloatingPointError('Reload forward logits NaN/Inf')
        checks.append({'id':row['id'],'image_grid_thw':detail['image_grid_thw'],'logits_shape':list(output.logits.shape),'finite':True})
        del output,batch
    save(run/'data_manifest.json',{'checkpoint_manifest':manifest,'independent_process_reload':True,'forward_check_ids':[r['id'] for r in checks]})
    save(run/'reload_checks.json',checks)
    # A private receipt is written separately; completed training metrics remain immutable.
    receipt={'reload_verified':True,'adapter_path':str(path),'checkpoint_manifest_sha256':sha(path.parent/'checkpoint_manifest.json'),
             'checks':checks,'reload_run':run.name,'no_new_original_model_inference':True}
    save(run/'reload_receipt.json',receipt)
    return {'reload_verified':True,'adapter_tensor_count':len(tensors),'generation_count':0,'image_forward_count':2,
      'independent_process':True,'adapter_path':str(path),'optimizer_steps':0}

def evaluate(cfg,run):
    from evaluate_vllm import evaluate as impl
    return impl(cfg,run)

def analyze(cfg,run):
    from rescore_latest import aggregate
    basepath,_,_=b0(cfg);base=rows(basepath/'rescored.jsonl');base_by={r['id']:r for r in base};runs=cfg.get('comparison_runs',{})
    if set(runs)!={'E1','E2'}:raise Blocked('Analysis requires completed E1 and E2 evaluation run paths')
    compared={};source_hashes={};deltas=[]
    for name,path in runs.items():
        path=Path(path);m=json.loads((path/'metrics.json').read_text())
        if m.get('status')!='completed' or m.get('MMMU_count')!=900:raise Blocked('Incomplete adapter MMMU evaluation:'+name)
        if m['B0_protocol_sha256']!=sha(basepath/'evaluation_protocol.json'):raise Blocked('Adapter/B0 protocol mismatch')
        results=rows(path/'responses.jsonl');subset=[r for r in results if r['dataset']=='MMMU']
        if {r['id'] for r in subset}!=set(base_by) or len(subset)!=900:raise Blocked('Adapter/B0 ID comparison differs')
        compared[name]={'MMMU':aggregate(subset),'checks':m.get('check_scores'),
          'by_subject':{s:aggregate([r for r in subset if r['subject']==s]) for s in sorted({r['subject'] for r in subset})}}
        compared[name]['delta_from_B0_pp']=compared[name]['MMMU']['correct_pct']-aggregate(base)['correct_pct']
        source_hashes[name]={'path':str(path),'responses_sha256':sha(path/'responses.jsonl'),'metrics_sha256':sha(path/'metrics.json')}
        for r in subset:deltas.append({'experiment':name,'id':r['id'],'subject':r['subject'],'B0_correct':base_by[r['id']]['correct'],
            'adapter_correct':r['correct'],'change':int(r['correct'])-int(base_by[r['id']]['correct'])})
    import csv
    with (run/'per_id_changes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(deltas[0]));w.writeheader();w.writerows(deltas)
    save(run/'data_manifest.json',{'B0_run':str(basepath),'B0_rescore_sha256':sha(basepath/'rescored.jsonl'),'adapter_evaluations':source_hashes})
    task_metrics=None
    if cfg.get('task_labels'):
        labels=rows(cfg['task_labels']);mapping={r['id']:r for r in labels}
        if len(mapping)!=len(labels) or not set(mapping).issubset(base_by):raise Blocked('Task labels must use unique known MMMU IDs')
        # Only explicitly reviewed task labels enter type metrics. Subjects are never substituted.
        reviewed={qid:r for qid,r in mapping.items() if r.get('review_status')=='reviewed' and isinstance(r.get('task_types'),list)}
        task_metrics={'reviewed_ids':len(reviewed),'total_ids':900,'not_official_MMMU_labels':True,'types':{}}
        for task in sorted({t for r in reviewed.values() for t in r['task_types']}):
            ids={qid for qid,r in reviewed.items() if task in r['task_types']};entry={}
            for name,results in [('B0',base)]+[(n,rows(Path(p)/'responses.jsonl')) for n,p in runs.items()]:
                sample=[r for r in results if r['id'] in ids];entry[name]=aggregate(sample)
            task_metrics['types'][task]=entry
    metrics={'B0':aggregate(base),'adapters':compared,'E2_minus_E1_pp':compared['E2']['MMMU']['correct_pct']-compared['E1']['MMMU']['correct_pct'],
      'task_type_metrics':task_metrics,'non_target_change':'unavailable unless explicit reviewed target/non-target labels supplied',
      'seed_count':1,'interpretation':'initial exploration; validation is a development set; not a replicated improvement claim',
      'scoring_policy_change_from_mixed_pp':json.loads((basepath/'metrics.json').read_text())['rescoring_delta_pp']}
    save(run/'comparison.json',metrics)
    lines=['# E1/E2 development comparison','', '| Model | Correct / 900 | Accuracy | Delta vs B0 (pp) | Parse failures | Length stops | Repetition stops |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,value in [('B0',metrics['B0'])]+[(n,v['MMMU']) for n,v in compared.items()]:
        delta=0 if name=='B0' else compared[name]['delta_from_B0_pp']
        lines.append(f"| {name} | {value['correct']}/900 | {value['correct_pct']:.2f}% | {delta:+.2f} | {value['parse_failures']} | {value['length_finish']} | {value['repetition_aborts']} |")
    lines+=['','Scoring-policy change versus the historical mixed score is separate from adapter effects.',
      'One seed; MMMU validation is a development set. Dataset heldout scores have no newly inferred base-model comparator.',
      'Task and non-target metrics require reviewed ID-level labels; subject names are not task labels.',
      'See comparison.json for subject and heldout counts; per_id_changes.csv preserves all paired decisions.','']
    (run/'comparison.md').write_text('\n'.join(lines))
    return metrics

def selftest(cfg,run):
    from scoring_latest_v2 import selftest as test
    result=test();save(run/'parser_tests.json',result)
    from test_pipeline import run_tests
    pipeline=run_tests(cfg,run);save(run/'pipeline_tests.json',pipeline)
    return {'parser':result,**pipeline}

def setup_eval(cfg,run):
    import shutil,subprocess,sys
    from evaluate_vllm import ESSENTIAL
    python=Path(cfg['eval_python']);directory=python.parent.parent;marker=directory/'.created_by_P04'
    if directory.exists() and not marker.is_file():
        raise Blocked('Evaluation environment already exists without P04 ownership marker; preserved. Configure an existing verified eval Python explicitly.')
    if shutil.disk_usage(ROOT).free<12*2**30 and not python.exists():raise Blocked('Less than12GiB free disk for separate vLLM/torch environment; no installation begun')
    save(run/'data_manifest.json',{'requirements_file':str(ROOT/'configs/P04/evaluation_requirements.txt'),
       'requirements_sha256':sha(ROOT/'configs/P04/evaluation_requirements.txt'),'existing_training_environment_preserved':True})
    if not directory.exists():
        subprocess.run([cfg['eval_base_python'],'-m','venv',str(directory)],check=True)
        marker.write_text('Created by P04; safe to resume package installation in this environment only.\n')
    command=[str(python),'-m','pip','install','--no-cache-dir','-r',str(ROOT/'configs/P04/evaluation_requirements.txt')]
    subprocess.run(command,check=True,stdout=sys.stdout,stderr=sys.stderr)
    subprocess.run([str(python),'-m','pip','check'],check=True,stdout=sys.stdout,stderr=sys.stderr)
    return {'eval_environment_installed':True,'eval_python':str(python),'requirements_sha256':sha(ROOT/'configs/P04/evaluation_requirements.txt'),
       'training_environment_modified':False,'GPU_inference_verified':False,'next':'Run pinned evaluation envcheck before any training'}
