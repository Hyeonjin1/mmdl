"""Fail-closed P04 E1/E2 data/evaluation contracts; no E3 dependency."""
import hashlib,json,re
from pathlib import Path
from records import ROOT,Blocked,save,sha,rows,jsonl
from rescore_latest import SNAP

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def b0(cfg):
    run=Path(cfg['b0_run']);metrics=json.loads((run/'metrics.json').read_text())
    if metrics.get('status')!='completed' or metrics.get('kind')!='CPU_saved_recent_vllm_rescore_not_new_inference':raise Blocked('B0 is not a completed recent-vLLM saved-response rescore')
    protocol=json.loads((run/'evaluation_protocol.json').read_text())
    if sha(run/'evaluation_protocol.json')!=metrics['evaluation_protocol_sha256']:raise Blocked('B0 evaluation protocol changed')
    for name in ['scoring_latest_v2.py','scoring_v2.py']:
        expected=protocol['scoring']['sha256' if name=='scoring_latest_v2.py' else 'normalization_dependency_sha256']
        if sha(ROOT/'code/p04'/name)!=expected:raise Blocked('Frozen B0 scorer changed:'+name)
    original=SNAP/'results/validation900_cot8192_guard_full/predictions.jsonl'
    if sha(original)!=protocol['source_predictions_sha256']:raise Blocked('Latest original response artifact changed')
    source=rows(original)
    if len(source)!=900 or len({r['question_id'] for r in source})!=900:raise Blocked('B0 ID contract is not900')
    return run,protocol,source

def image_manifest(cfg):
    path=Path(cfg['mmmu_manifest']);_,protocol,source=b0(cfg)
    if not path.is_file():raise Blocked('Recent MMMU image acquisition not complete; wait for the host run')
    if sha(path)!=protocol['dataset_manifest_sha256']:raise Blocked('Recent MMMU image export does not match historical manifest checksum')
    from PIL import Image
    data=rows(path);lookup={r['index']:r for r in data}
    if set(lookup)!={r['question_id'] for r in source} or len(data)!=900:raise Blocked('MMMU image IDs differ from recent900')
    for pred in source:
        r=lookup[pred['question_id']];a=pred['annotation']
        for key in ['question','answer','question_type','category']:
            if r[key]!=a[key]:raise Blocked('MMMU annotation mismatch:'+r['index'])
        options={k:v for k,v in r.items() if len(k)==1 and k in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
        expected={k:a[k] for k in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if a.get(k) is not None}
        if options!=expected:raise Blocked('MMMU option mapping changed:'+r['index'])
        if len(r['image_path'])!=len(a['image_path']) or len(r['image_path'])!=len(r['image_sha256']):raise Blocked('MMMU image count changed')
        for n,(name,expected_hash) in enumerate(zip(r['image_path'],r['image_sha256']),1):
            image=(path.parent/name).resolve()
            if not image.is_relative_to(path.parent.resolve()) or not image.name.endswith('_'+str(n)+'.png'):raise Blocked('MMMU image order/path invalid')
            if not image.is_file() or sha(image)!=expected_hash:raise Blocked('MMMU image integrity failure:'+str(image))
            with Image.open(image) as im:im.verify()
    return path,data

def ready(cfg,run):
    from data import image_info,textkey
    source_dir=Path(cfg['source_processed_dir']);original=json.loads((source_dir/'manifest.json').read_text())
    mpath,evalrows=image_manifest(cfg);selected={}
    for name in ['E1','E2','ChartQA_check','TabMWP_check']:
        p=source_dir/(name+'.jsonl')
        if sha(p)!=original['files'][name]['sha256']:raise Blocked('Acquired dataset changed:'+name)
        selected[name]=rows(p)
    if [len(selected[k]) for k in selected]!=[1000,1000,100,100]:raise Blocked('Expected E1/E2 1000+1000 and checks100+100')
    if len({r['id'] for r in selected['E1']})!=1000 or len({r['id'] for r in selected['E2']})!=1000:raise Blocked('Duplicate QA IDs')
    if {r['id'] for r in selected['E2'] if r['source']=='ChartQA'}-{r['id'] for r in selected['E1']}:raise Blocked('E2 ChartQA is not E1 subset')
    if sum(r['source']=='ChartQA' for r in selected['E2'])!=500 or sum(r['source']=='TabMWP' for r in selected['E2'])!=500:raise Blocked('E2 source counts differ')
    training=selected['E1']+selected['E2'];checks=selected['ChartQA_check']+selected['TabMWP_check']
    if {r['group_id'] for r in training}&{r['group_id'] for r in checks}:raise Blocked('Training/check original group overlap')
    if {r['pixel_sha256'] for r in training}&{r['pixel_sha256'] for r in checks}:raise Blocked('Training/check identical image overlap')
    evalquestions={textkey(r['question']) for r in evalrows};pixels=set();perceptual=[]
    for r in evalrows:
        for name in r['image_path']:
            info=image_info(str(mpath.parent/name));pixels.add(info['pixel_sha256']);perceptual.append((r['index'],info['dhash']))
    overlaps=[];suspects=[]
    for r in {r['id']:r for r in training}.values():
        if textkey(r['question']) in evalquestions or r['pixel_sha256'] in pixels:overlaps.append(r['id'])
        close=[qid for qid,h in perceptual if (r['dhash']^h).bit_count()<=2]
        if close:suspects.append({'id':r['id'],'MMMU_candidate_ids':close,'review_status':'unreviewed; training blocked'})
    save(run/'overlap_checks.json',{'confirmed':overlaps,'perceptual_candidates':suspects,'semantic_overlap_exhaustively_checked':False})
    if overlaps or suspects:raise Blocked('MMMU overlap confirmed or near-image review pending; no silent row exclusion/replacement')
    target=Path(cfg['processed_dir']);target.mkdir(parents=True,exist_ok=True);files={}
    for name,data in selected.items():
        path=target/(name+'.jsonl');content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data)
        if path.exists() and path.read_text()!=content:raise Blocked('Existing processed E1/E2 differs; preserve it')
        if not path.exists():path.write_text(content)
        files[name]={'path':str(path),'sha256':sha(path),'count':len(data)}
    manifest={'processing_version':'P04_E12_v1','ready_for_training':True,'E3_required':False,'files':files,
       'source_manifest_sha256':sha(source_dir/'manifest.json'),'sources':original['sources'],
       'MMMU_manifest_sha256':sha(mpath),'MMMU_questions_checked':900,'MMMU_image_overlap_checked':True,
       'train_check_original_groups_disjoint':True,'seed':3407,'B0_protocol_sha256':sha(Path(cfg['b0_run'])/'evaluation_protocol.json'),
       'manual_sample_review':'4 source samples visually reviewed in prior CPU run; not full reasoning verification',
       'task_type_labels':cfg.get('task_labels'),'semantic_overlap_exhaustively_checked':False}
    dest=target/'manifest.json'
    if dest.exists() and json.loads(dest.read_text())!=manifest:raise Blocked('Ready data contract changed; use a new processed version')
    save(dest,manifest);save(run/'data_manifest.json',manifest)
    return {'ready_for_training':True,'E1_count':1000,'E2_count':1000,'check_count':200,'E3_required':False,'MMMU_overlap_checks_passed':True}

def training_rows(cfg,run,smoke=False):
    path=Path(cfg['processed_dir'])/'manifest.json'
    if not path.exists():raise Blocked('Run recorded E1/E2 readiness stage first')
    manifest=json.loads(path.read_text());mpath,_=image_manifest(cfg)
    if not manifest.get('ready_for_training') or manifest['MMMU_manifest_sha256']!=sha(mpath):raise Blocked('Ready data manifest invalid')
    b0path,_,_=b0(cfg)
    if manifest['B0_protocol_sha256']!=sha(b0path/'evaluation_protocol.json'):raise Blocked('B0 protocol changed')
    name='E2' if smoke else cfg['experiment']
    file=Path(manifest['files'][name]['path'])
    if sha(file)!=manifest['files'][name]['sha256']:raise Blocked('Training data file changed')
    data=rows(file)
    if smoke:
        #4 ChartQA and4 TabMWP =>8 exposures,2 true optimizer updates; separate adapter.
        data=[r for r in data if r['source']=='ChartQA'][:4]+[r for r in data if r['source']=='TabMWP'][:4]
    if len(data)!=(8 if smoke else 1000):raise Blocked('Actual training QA count differs')
    save(run/'data_manifest.json',{'ready_manifest_sha256':sha(path),'ready_manifest':manifest,'training_ids':[r['id'] for r in data],
       'training_count':len(data),'source_file_sha256':sha(file),'smoke':smoke,'evaluation_data_in_training':False})
    return data,manifest

def checkpoint(cfg):
    path=Path(cfg.get('adapter_path') or '')
    if not (path/'adapter_model.safetensors').is_file():raise Blocked('Adapter checkpoint missing')
    parent=path.parent;metrics=json.loads((parent/'metrics.json').read_text());manifest=json.loads((parent/'checkpoint_manifest.json').read_text())
    if metrics.get('status')!='completed' or not metrics.get('adapter_saved'):raise Blocked('Adapter not from completed training')
    if manifest['base_revision']!=cfg['model_revision'] or manifest['base_model_id']!=cfg['model_id']:raise Blocked('Adapter original base identity differs')
    if sha(parent/'data_manifest.json')!=manifest['data_manifest_sha256']:raise Blocked('Checkpoint training data manifest changed')
    for name,h in manifest['files_sha256'].items():
        if sha(path/name)!=h:raise Blocked('Adapter artifact changed:'+name)
    return path,manifest,metrics
