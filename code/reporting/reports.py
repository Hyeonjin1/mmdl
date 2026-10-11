"""Public allowlist exporter and account-checked, non-force GitHub publisher."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
STAGING=ROOT/'public_reports'
CLONE=ROOT.parent/'mmdl-reports'
REPO='Hyeonjin1/mmdl'
CONFIG_KEYS='model_id model_revision seed train_size check_size world_size bf16 lora_rank lora_alpha lora_dropout micro_batch gradient_accumulation optimizer_steps optimizer learning_rate weight_decay warmup_steps scheduler max_grad_norm max_sequence_length min_pixels max_pixels gradient_checkpointing packing assistant_only_loss generation_max_new_tokens generation_do_sample generation_parser stage'.split()
METRIC_KEYS='status optimizer_steps tests_passed tests_failed rows_checked max_total_tokens min_supervised_tokens question_exposures unique_questions_exposed epochs_equivalent input_tokens supervised_tokens peak_allocated_bytes peak_reserved_bytes training_seconds first_loss last_loss trainable_parameter_count adapter_saved reload_verified generation_count elapsed_seconds wall_seconds started_utc ended_utc verified_optimizer_steps checkpoint_hashes_verified adapter_tensor_count adapter_parameters adapter_B_nonzero peak_allocated_gib peak_reserved_gib'.split()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def public_files():
    result=[]
    for path in sorted(STAGING.rglob('*')):
        if not path.is_file(): continue
        rel=path.relative_to(STAGING)
        allowed=str(rel) in ('INDEX.md','INDEX.json') or (
            len(rel.parts)==3 and rel.parts[0]=='runs' and rel.parts[1].startswith(('P03_','P04_'))
            and rel.name in ('experiment.md','config_resolved.json','metrics.json','public_predictions.jsonl'))
        if not allowed or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise RuntimeError('Unexpected file in public staging: '+str(rel))
        result.append(path)
    return result

def scan(path):
    text=path.read_text()
    for forbidden in ['/home/','/tmp/','response_raw','rendered_prompt','original_id','image_sha256','per_question','train_human.json#','BEGIN PRIVATE KEY','ghp_','github_pat_']:
        if forbidden in text: raise RuntimeError('Public allowlist scan failed: '+path.name)
    limit=1024*1024 if path.name=='public_predictions.jsonl' else 100000
    if path.stat().st_size>limit: raise RuntimeError('Public file too large')
    if path.name=='public_predictions.jsonl':
        fields={'id','subject','question_type','extracted_answer','correct','extraction_method','finish_reason','input_tokens','output_tokens'}
        for line in text.splitlines():
            row=json.loads(line)
            if set(row)!=fields: raise RuntimeError('Public prediction fields differ from allowlist')
            if not re.fullmatch(r'validation_[A-Za-z0-9_]+',row['id']): raise RuntimeError('Unexpected public question ID')
            if row['extracted_answer'] is not None and len(str(row['extracted_answer']))>512: raise RuntimeError('Public extracted answer too long')

def export(run_id):
    if run_id.startswith('P04_'): return export_p04(run_id)
    if Path(run_id).name!=run_id or not run_id.startswith('P03_'): raise ValueError('Invalid run ID')
    run=ROOT/'runs'/run_id
    for name in ['experiment.md','config_resolved.json','metrics.json']:
        if not (run/name).is_file(): raise RuntimeError('Missing required source record: '+name)
    config=json.loads((run/'config_resolved.json').read_text())
    metrics=json.loads((run/'metrics.json').read_text())
    if metrics['status'] not in ['completed','oom','error','blocked','interrupted']: raise RuntimeError('Run not terminal')
    c={k:config[k] for k in CONFIG_KEYS if k in config}
    m={k:metrics[k] for k in METRIC_KEYS if k in metrics}
    if 'scores' in metrics:
        m['scores']={k:{q:v[q] for q in ['correct','total']} for k,v in metrics['scores'].items()}
    if 'generation_checks' in metrics:
        m['generation_checks']={k:{q:v[q] for q in ['correct','total','output_cap_reached','parse_failure','empty_response']} for k,v in metrics['generation_checks'].items()}
    if metrics['status'] in ['oom','error','blocked','interrupted']:
        m['error_details']='See private local records; raw errors are excluded from public reports.'
    out=STAGING/'runs'/run_id
    # Refuse to silently change a previously exported result.
    for name,value in [('config_resolved.json',c),('metrics.json',m)]:
        dest=out/name
        if dest.exists() and json.loads(dest.read_text())!=value: raise RuntimeError('Existing public result differs: '+run_id)
        write(dest,value)
    md=f'''# {run_id}

Purpose: P03 ChartQA gold-answer SFT pipeline validation. This is not KD or a performance-improvement claim.

Stage: {c['stage']}; terminal status: {m['status']}.

Actual settings: see [config_resolved.json](config_resolved.json).
Results: see [metrics.json](metrics.json).

Data: official ChartQA train_human; fixed seed3407;64 distinct training charts and16 different check charts; one original QA per chart. Source revision044eabfc306abfe9340c5741f0093aefc5973d06. Evaluation datasets and corrected evaluation solutions are excluded from training.

Loss covers assistant answers and necessary end tokens only. Language attention/MLP LoRA; vision encoder and merger frozen. For completed20-update training,80 question exposures comprise48 questions seen once and16 seen twice, equivalent to1.25 epochs.

Limitations: base/adapter16-question generation is a pipeline check. Full-response strict strip/casefold matching and64-token generation cap affect the scores. In the completed P03 generation check, base14/16 responses reached the cap; adapter0/16. Cap reached is not the same as missing a final answer. No MMMU evaluation or improvement claim. Prior563/900 combines different judgment rules and is not a matched baseline.

Next step: curate separate training data and validate teacher quality before a separately authorized KD run. Teacher calls and KD training were not started.

Public scope: summary settings and metrics only. Data, images, checkpoints, responses, commands, credentials and large logs are excluded. Private source experiment SHA256: {hashlib.sha256((run/'experiment.md').read_bytes()).hexdigest()}.
'''
    (out/'experiment.md').write_text(md)
    entries=[]
    for folder in sorted((STAGING/'runs').iterdir()):
        metrics_path=folder/'metrics.json'
        if not metrics_path.is_file(): continue
        mm=json.loads(metrics_path.read_text());cc=json.loads((folder/'config_resolved.json').read_text())
        entries.append(dict(run_id=folder.name,stage=cc['stage'],status=mm['status'],optimizer_steps=mm.get('optimizer_steps',0),report=f'runs/{folder.name}/experiment.md'))
    write(STAGING/'INDEX.json',dict(repository=REPO,scope='P03 pipeline validation; no improvement claim',runs=entries))
    lines=['# MMDL experiment reports','','Public summaries only. P03 is gold-answer SFT pipeline validation, not KD or a performance improvement claim.','','| Run | Stage | Status | Updates |','|---|---|---|---:|']
    lines += [f"| [{r['run_id']}]({r['report']}) | {r['stage']} | {r['status']} | {r['optimizer_steps']} |" for r in entries]
    (STAGING/'INDEX.md').write_text('\n'.join(lines)+'\n')
    for p in public_files(): scan(p)
    return out

def export_p04(run_id):
    if Path(run_id).name!=run_id: raise ValueError('Invalid run ID')
    run=ROOT/'runs'/run_id
    for name in ['experiment.md','config_resolved.json','metrics.json']:
        if not (run/name).is_file(): raise RuntimeError('Missing required source record: '+name)
    config=json.loads((run/'config_resolved.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
    if metrics['status'] not in ['completed','oom','error','blocked','interrupted']:raise RuntimeError('Run not terminal')
    c={k:config[k] for k in CONFIG_KEYS if k in config}
    c['purpose']='P04 original-answer SFT/data/evaluation preparation; no teacher or KD'
    protocol_path=run/'evaluation_protocol.json'
    if not protocol_path.is_file() and config.get('b0_run'):
        protocol_path=Path(config['b0_run'])/'evaluation_protocol.json'
    if protocol_path.is_file():
        protocol=json.loads(protocol_path.read_text())
        c.update(model_id=protocol['model_id'],model_revision=protocol['model_revision'],
          evaluation_settings=protocol['settings'],prompt_protocol=protocol['prompt_protocol'],
          repetition_policy=protocol['repetition_policy'],image_order=protocol['image_order'],
          scoring=protocol['scoring'],backend='vllm==0.11.2',original_model_inference=False)
    extra='kind audit_only unique_rows_checked E1_rows E2_rows heldout_rows all_assistant_masks_verified truncation real_padding gpu_forward_backward_verified MMMU_image_overlap_verified E3_required ready_for_training source_correct_mixed source_total source_accuracy_pct_mixed changed_correctness_from_mixed mixed_correct_to_new_incorrect mixed_incorrect_to_new_correct rescoring_delta_pp evaluation_protocol_sha256 MMMU_count B0_protocol_sha256 smoke adapter_reload_verified original_model_inference_requests actual_warmup_updates original_initialization peak_device_memory_used_mib parent_allocator_is_not_vllm_worker'.split()
    m={k:metrics[k] for k in METRIC_KEYS+extra if k in metrics}
    for key in ['overall','by_subject','by_question_type','check_scores']:
        if key in metrics:m[key]=metrics[key]
    for key in ['B0','adapters','E2_minus_E1_pp','task_type_metrics','non_target_change','seed_count','interpretation','scoring_policy_change_from_mixed_pp']:
        if key in metrics:m[key]=metrics[key]
    if metrics['status']!='completed':m['error_details']='Private record contains the exact error; dependent execution stopped.'
    out=STAGING/'runs'/run_id
    for name,value in [('config_resolved.json',c),('metrics.json',m)]:
        dest=out/name
        if dest.exists() and json.loads(dest.read_text())!=value:raise RuntimeError('Existing public result differs:'+run_id)
        write(dest,value)
    overall=metrics.get('overall');result='Actual completed operations are recorded in the stage settings and metrics. Original-model responses were not regenerated.'
    if overall and metrics.get('kind')=='CPU_saved_recent_vllm_rescore_not_new_inference':
        result=(f"Uniform saved-response B0: {overall['correct']}/{overall['total']} ({overall['correct_pct']:.2f}%). "
          f"Prior mixed judging:563/900(62.56%); rescoring delta {metrics['rescoring_delta_pp']:+.2f} percentage points. "
          f"Extraction failures:{overall['parse_failures']}; length cap:{overall['output_cap_reached']}; repetition abort:{overall['repetition_aborts']}. "
          'This difference is scoring-policy change on identical responses, not training improvement. Initial493/900 responses were not used.')
    elif overall:
        result=(f"Adapter MMMU: {overall['correct']}/{overall['total']} ({overall['correct_pct']:.2f}%). "
          f"Extraction failures:{overall['parse_failures']}; length cap:{overall['output_cap_reached']}; repetition abort:{overall['repetition_aborts']}. "
          'Smoke results are pipeline checks; full single-seed results are development comparisons. B0 is reused saved-response rescoring; original model was not inferred again.')
    elif metrics.get('adapter_saved'):
        result=(f"Original model initialized independently; completed {metrics['optimizer_steps']} updates and {metrics.get('question_exposures')} question exposures. "
          f"Adapter saved; separate-process reload status recorded separately. Smoke:{metrics.get('smoke')}. No teacher/KD.")
    md=f'''# {run_id}

Purpose: P04 recorded preparation and latest-vLLM evaluation audit. No teacher/KD.

Stage: {c.get('stage')}; status: {m['status']}.

{result}

Actual settings and summary metrics: [config_resolved.json](config_resolved.json), [metrics.json](metrics.json).

Limitations: MMMU validation is a development comparison set. Exact answer and unit-preserving scoring can miss semantic equivalents. No reasoning verification, official score or training improvement claim. Work-type labels are unavailable; subjects are not task types. P03's64-token check is separate from latest-vLLM8192-token evaluation. GPU and matching original images must be verified before dependent E1/E2 execution. E3/PGPS9K availability does not gate E1/E2. Failed prerequisites stop dependent runs.

Public scope: settings, aggregates, this report and optional ID-level judgments only. No raw responses, questions, gold answers, images, original data, checkpoints, logs or credentials. Private source experiment SHA256: {hashlib.sha256((run/'experiment.md').read_bytes()).hexdigest()}.
'''
    dest=out/'experiment.md'
    if dest.exists() and dest.read_text()!=md:raise RuntimeError('Existing public report differs:'+run_id)
    dest.write_text(md)
    source=run/'public_predictions.jsonl'
    if source.is_file():
        dest=out/source.name
        if dest.exists() and dest.read_bytes()!=source.read_bytes():raise RuntimeError('Existing public judgments differ:'+run_id)
        shutil.copyfile(source,dest)
    entries=[]
    for folder in sorted((STAGING/'runs').iterdir()):
        if not (folder/'metrics.json').is_file():continue
        mm=json.loads((folder/'metrics.json').read_text());cc=json.loads((folder/'config_resolved.json').read_text())
        entries.append(dict(run_id=folder.name,stage=cc['stage'],status=mm['status'],optimizer_steps=mm.get('optimizer_steps',0),report=f'runs/{folder.name}/experiment.md'))
    write(STAGING/'INDEX.json',dict(repository=REPO,scope='P03 pipeline checks and P04 data/evaluation records; no improvement claim',runs=entries))
    lines=['# MMDL experiment reports','','Sanitized experiment summaries. Saved-response rescoring and training effects are separate.','','| Run | Stage | Status | Updates |','|---|---|---|---:|']
    lines.extend(f"| [{r['run_id']}]({r['report']}) | {r['stage']} | {r['status']} | {r['optimizer_steps']} |" for r in entries)
    (STAGING/'INDEX.md').write_text('\n'.join(lines)+'\n')
    for path in public_files():scan(path)
    return out

def command(args,**kwargs):
    result=subprocess.run(args,capture_output=True,text=True,timeout=90,**kwargs)
    if result.returncode:
        detail=result.stderr.strip() or result.stdout.strip()
        detail=re.sub(r'(gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)', '[REDACTED]', detail)
        detail=re.sub(r'(?i)(authorization[:=]\s*|token[:=]\s*)\S+', r'\1[REDACTED]', detail)
        raise RuntimeError('Command failed ('+str(result.returncode)+'): '+args[0]+' '+args[1]+'; '+detail[:4000])
    return result.stdout.strip()

def publish():
    if not shutil.which('gh'): raise RuntimeError('GitHub CLI gh is absent; install it and authenticate Hyeonjin1')
    login=command(['gh','api','user','--jq','.login'])
    if login!='Hyeonjin1': raise RuntimeError('CLI authenticated account is not Hyeonjin1')
    lookup=subprocess.run(['gh','api','repos/'+REPO],capture_output=True,text=True,timeout=45)
    if lookup.returncode:
        # An authentication/network error must not be mistaken for an absent repo.
        if '404' not in lookup.stderr: raise RuntimeError('Repository lookup failed; absence not confirmed')
        command(['gh','repo','create',REPO,'--public','--description','Public experiment summaries; no datasets, responses or checkpoints'])
    metadata=json.loads(command(['gh','api','repos/'+REPO]))
    if metadata.get('private'): raise RuntimeError('Existing repository is private; visibility not changed automatically')
    if metadata.get('full_name')!=REPO or not metadata.get('permissions',{}).get('push'):
        raise RuntimeError('Authenticated CLI account lacks confirmed target repository push permission')
    command(['gh','auth','setup-git'])
    if not CLONE.exists(): command(['gh','repo','clone',REPO,str(CLONE)])
    actual=command(['git','-C',str(CLONE),'remote','get-url','origin'])
    if actual.rstrip('/') not in ['https://github.com/'+REPO+'.git','https://github.com/'+REPO,'git@github.com:'+REPO+'.git']:
        raise RuntimeError('Unexpected report repository origin')
    if command(['git','-C',str(CLONE),'status','--porcelain']): raise RuntimeError('Report clone has existing local changes; preserved without publishing')
    refs=command(['git','-C',str(CLONE),'ls-remote','--heads','origin'])
    if refs: command(['git','-C',str(CLONE),'pull','--ff-only'])
    else:
        head=subprocess.run(['git','-C',str(CLONE),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
        if head.returncode:
            command(['git','-C',str(CLONE),'symbolic-ref','HEAD','refs/heads/main'])
    if command(['git','-C',str(CLONE),'branch','--show-current'])!='main':
        raise RuntimeError('Report clone is not on main; existing branch preserved')
    files=public_files()
    for p in files:
        scan(p);dest=CLONE/p.relative_to(STAGING)
        if dest.exists() and dest.parent.parent.name=='runs' and dest.read_bytes()!=p.read_bytes():
            raise RuntimeError('Published run differs; no overwrite')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    names=[str(p.relative_to(STAGING)) for p in files]
    command(['git','-C',str(CLONE),'add','--',*names])
    if command(['git','-C',str(CLONE),'diff','--cached','--name-only']):
        command(['git','-C',str(CLONE),'-c','user.name=Hyeonjin1','-c','user.email=101498696+Hyeonjin1@users.noreply.github.com','commit','-m','Publish sanitized experiment reports'])
    command(['git','-C',str(CLONE),'push','-u','origin','HEAD'])
    commit=command(['git','-C',str(CLONE),'rev-parse','HEAD'])
    remote=command(['gh','api','repos/'+REPO+'/git/ref/heads/main','--jq','.object.sha'])
    if remote!=commit: raise RuntimeError('Remote main differs from the pushed commit; verification failed')
    tree_sha=command(['gh','api','repos/'+REPO+'/git/commits/'+commit,'--jq','.tree.sha'])
    tree=json.loads(command(['gh','api','repos/'+REPO+'/git/trees/'+tree_sha+'?recursive=1']))
    if tree.get('truncated'): raise RuntimeError('Remote tree verification was truncated')
    blobs={entry['path']:entry['sha'] for entry in tree['tree'] if entry['type']=='blob'}
    for file in files:
        data=file.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blobs.get(str(file.relative_to(STAGING)))!=blob:
            raise RuntimeError('Remote public file hash differs: '+file.name)
    import base64
    index=json.loads(command(['gh','api','repos/'+REPO+'/contents/INDEX.md?ref='+commit]))
    if base64.b64decode(index['content'])!=(STAGING/'INDEX.md').read_bytes():
        raise RuntimeError('Remote INDEX content verification failed')
    return commit

def record_batch(status):
    # The last batch call publishes all staged runs, not only its --run-id.
    # Preserve earlier records and make the shared publication result explicit.
    entries=json.loads((STAGING/'INDEX.json').read_text())['runs']
    for entry in entries:
        run_id=entry['run_id']
        if Path(run_id).name!=run_id or not run_id.startswith(('P03_','P04_')): raise ValueError('Invalid staged run')
        folder=ROOT/'runs'/run_id/'publication';folder.mkdir(exist_ok=True)
        path=folder/'status.json'
        if path.exists():
            with (folder/'batch_history.jsonl').open('a') as history:
                history.write(json.dumps(json.loads(path.read_text()))+'\n')
        result=dict(status,run_id=run_id,batch_trigger_run_id=status['run_id'])
        if status.get('publication_status')=='completed':
            result.update(remote_verified=True,index_url='https://github.com/'+REPO+'/blob/main/INDEX.md')
        write(path,result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--publish',action='store_true');args=p.parse_args()
    record=ROOT/'runs'/args.run_id/'publication'
    record.mkdir(exist_ok=True)
    prior=record/'status.json'
    if prior.exists():
        with (record/'history.jsonl').open('a') as history: history.write(prior.read_text().strip().replace('\n','')+'\n')
    status=dict(run_id=args.run_id,recorded_utc=datetime.now(timezone.utc).isoformat(),repository=REPO,export_status='pending',publication_status='not_requested')
    try:
        export(args.run_id);status['export_status']='completed'
        if args.publish:
            status['publication_status']='running';write(record/'status.json',status)
            status['commit_sha']=publish();status['publication_status']='completed'
    except Exception as exc:
        status['error']=str(exc);status['publication_status']='failed' if status['export_status']=='completed' else 'not_attempted'
        write(record/'status.json',status)
        if args.publish and status['export_status']=='completed': record_batch(status)
        print('Report export/publication failed; experiment metrics unchanged.',file=sys.stderr)
        return 1
    write(record/'status.json',status)
    if args.publish: record_batch(status)
    print(json.dumps(status));return 0

if __name__=='__main__':sys.exit(main())
