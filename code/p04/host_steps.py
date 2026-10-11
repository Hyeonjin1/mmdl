"""Recorded host GPU probe and exact recent-MMMU image acquisition. No model inference."""
import argparse,datetime,hashlib,io,json,os,re,shlex,shutil,subprocess,sys,traceback
from pathlib import Path
from records import ROOT,save,environment,provenance,index,now,sha,Blocked
from rescore_latest import SNAP,read

DATA_REV='98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'

def create(stage):
    rid='P04_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+stage
    run=ROOT/'runs'/rid;run.mkdir();(run/'logs').mkdir()
    cfg={'stage':stage,'seed':3407,'model_id':'Qwen/Qwen3-VL-4B-Instruct',
         'model_revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17','dataset_revision':DATA_REV,
         'original_model_inference_allowed':False,'existing_processes':'read-only; never terminate'}
    save(run/'config_resolved.json',cfg);save(run/'environment.json',environment());save(run/'code_provenance.json',provenance())
    save(run/'data_manifest.json',{'status':'pending'});save(run/'metrics.json',{'status':'created','optimizer_steps':0})
    (run/'experiment.md').write_text('# '+rid+'\n\nPurpose: '+stage+'; preserve all existing work.\n')
    (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/user/MMDL\n'+shlex.join([sys.executable,str(Path(__file__).resolve()),stage])+'\n')
    index();return run

def gpu(run):
    commands=[['nvidia-smi'],['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],
              ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']]
    outputs=[]
    for cmd in commands:
        p=subprocess.run(cmd,capture_output=True,text=True);outputs.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    info=Path('/proc/driver/nvidia/gpus/0000:01:00.0/information')
    evidence={'queries':outputs,'proc_driver_info':info.read_text() if info.exists() else None,
              'nvidia_devices':[str(p) for p in Path('/dev').glob('nvidia*')],
              'scope':'current process environment only; host health cannot be inferred from sandbox failure'}
    import torch
    evidence.update(cuda_available=torch.cuda.is_available(),device_count=torch.cuda.device_count(),torch=torch.__version__,cuda_runtime=torch.version.cuda)
    try:
        x=torch.tensor([2.,3.],device='cuda:0');value=(x*x).sum().item();torch.cuda.synchronize()
        assert value==13.;evidence.update(cuda_tensor_result=value,device_name=torch.cuda.get_device_name(0),tensor_test='passed')
    except Exception as e:evidence.update(tensor_test='failed',tensor_error=repr(e))
    save(run/'gpu_probe.json',evidence)
    if outputs[0]['returncode'] or evidence['tensor_test']!='passed':
        raise Blocked('GPU probe failed in current environment; execute scripts/check_p04_host_gpu.sh from the host terminal. Host GPU failure is not established.')
    return {'gpu_probe':'passed','cuda_tensor_result':13.,'device':evidence['device_name'],'processes_stopped':0}

def images(run):
    from PIL import Image
    source=SNAP/'results/validation900_cot8192_guard_full/predictions.jsonl';preds=read(source)
    rc=json.loads((source.parent/'run_config.json').read_text());target=ROOT/'datasets/raw/MMMU_latest_vllm'
    target.mkdir(parents=True,exist_ok=True);imgdir=target/'images';imgdir.mkdir(exist_ok=True)
    for candidate in [Path('/home/user/mmdl/MMDL-official-vllm/data/local_validation/validation.jsonl'),target/'validation.jsonl']:
        if candidate.is_file() and sha(candidate)==rc['dataset_sha256']:
            # Original manifest and images are verified individually, including PNG decoding.
            available=read(candidate)
            for row in available:
                for name,digest in zip(row['image_path'],row['image_sha256']):
                    path=candidate.parent/name
                    if not path.exists() or sha(path)!=digest:raise Blocked('Original image missing/hash mismatch: '+str(path))
                    with Image.open(path) as im:im.verify()
            save(run/'data_manifest.json',{'ready':True,'path':str(candidate),'sha256':sha(candidate),'count':900,'source':'verified original export'})
            return {'image_dataset_ready':True,'count':900,'manifest':str(candidate),'original_manifest_hash_matched':True}
    # Fetch only official validation parquet files at the historical revision; no current-main fallback.
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi();subjects=sorted({r['annotation']['category'] for r in preds})
    existing=Path(rc['dataset_provenance']['source_path'])
    data={};downloaded=[]
    if (existing/'dataset_dict.json').is_file():
        from datasets import load_from_disk
        data=load_from_disk(str(existing))
    else:
        # Fast TLS-verified reachability check prevents prolonged hub retries when sandbox DNS is absent.
        import urllib.request
        with urllib.request.urlopen('https://huggingface.co/api/datasets/MMMU/MMMU/revision/'+DATA_REV,timeout=20) as f:
            metadata=json.load(f)
        siblings=metadata.get('siblings',[])
        files=[x['rfilename'] for x in siblings if x['rfilename'].split('/')[0] in subjects and '/validation' in x['rfilename'] and x['rfilename'].endswith('.parquet')]
        if len(files)!=30:raise Blocked('Expected official 30 validation parquet files at pinned revision, found '+str(len(files)))
        if shutil.disk_usage(target).free<4*2**30:raise Blocked('Less than4GiB free disk reserve before image download')
        import pyarrow.parquet as pq
        for name in files:
            path=Path(hf_hub_download(repo_id='MMMU/MMMU',repo_type='dataset',revision=DATA_REV,filename=name))
            downloaded.append({'source':name,'cache_path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
            data[name.split('/')[0]]=pq.read_table(path).to_pylist()
    byid={}
    for subject in subjects:
        for sample in data[subject]:byid[sample['id']]=sample
    rows=[];image_count=0;checks=[]
    for pred in preds:
        ann=pred['annotation'];sample=byid.get(pred['question_id'])
        if sample is None:raise Blocked('Missing historical ID:'+pred['question_id'])
        if any(sample[k]!=ann[k] for k in ['question','answer','question_type']):raise Blocked('Question/gold/type mismatch at pinned revision:'+pred['question_id'])
        import ast
        options=ast.literal_eval(sample['options']) if isinstance(sample['options'],str) else sample['options']
        if ann['question_type']=='multiple-choice' and options!=[ann[k] for k in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if ann.get(k) is not None]:
            raise Blocked('Options mismatch:'+pred['question_id'])
        row={k:ann[k] for k in ['index','question','answer','split','category','question_type']}
        if ann['question_type']=='multiple-choice':row.update({chr(65+i):v for i,v in enumerate(options)})
        paths=[];hashes=[];numbers=[]
        for n in range(1,8):
            image=sample.get('image_'+str(n))
            if image is None:continue
            if isinstance(image,dict):
                image=Image.open(io.BytesIO(image['bytes'])) if image.get('bytes') else Image.open(image['path'])
            dest=imgdir/(pred['question_id']+'_'+str(n)+'.png');buf=io.BytesIO();image.convert('RGB').save(buf,format='PNG');new=buf.getvalue()
            if dest.exists() and dest.read_bytes()!=new:raise Blocked('Existing image differs; preserve it and use a new raw version:'+str(dest))
            if not dest.exists():dest.write_bytes(new)
            with Image.open(dest) as im:im.load();size=list(im.size)
            paths.append('images/'+dest.name);hashes.append(sha(dest));numbers.append(n);image_count+=1
            checks.append({'id':pred['question_id'],'image_number':n,'sha256':sha(dest),'size':size,'decoded':True})
        if len(paths)!=len(ann['image_path']) or numbers!=list(range(1,len(paths)+1)):raise Blocked('Image count/order mismatch:'+pred['question_id'])
        refs=set(map(int,re.findall(r'<image\s+(\d+)>',ann['question']+'\n'+'\n'.join(options if ann['question_type']=='multiple-choice' else []))))
        if not refs.issubset(numbers):raise Blocked('Missing referenced image:'+pred['question_id'])
        row.update(image_path=paths,image_sha256=hashes);rows.append(row)
    # Original export sorted subjects then original row order. Exact original hash is the strongest identity test.
    rows.sort(key=lambda r:(r['category'],int(r['index'].rsplit('_',1)[1])))
    content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode();manifest=target/'validation.jsonl'
    if manifest.exists() and manifest.read_bytes()!=content:raise Blocked('Existing manifest differs; no overwrite')
    if not manifest.exists():manifest.write_bytes(content)
    matched=sha(manifest)==rc['dataset_sha256']
    provenance={'source':'MMMU/MMMU official validation','revision':DATA_REV,'count':len(rows),'image_count':image_count,
      'manifest_sha256':sha(manifest),'historical_manifest_sha256':rc['dataset_sha256'],'original_manifest_hash_matched':matched,
      'annotation_matches_recent_vllm':True,'image_policy':'RGB PNG, ascending original image_N; all images before text',
      'files':downloaded,'image_checks':checks,'license':'Apache-2.0 dataset card; underlying source rights not independently reviewed',
      'ready':matched,'acquired_utc':now()}
    save(target/'provenance.json',provenance);save(run/'data_manifest.json',provenance)
    if not matched:raise Blocked('900 QA matched but historical export manifest hash differs. Possible PNG encoding/revision difference; image identity must be reviewed before claiming identical evaluation.')
    return {'image_dataset_ready':True,'count':900,'images':image_count,'manifest':str(manifest),'original_manifest_hash_matched':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['host_gpu_probe','acquire_recent_mmmu']);a=p.parse_args();run=create(a.stage)
    try:
        save(run/'metrics.json',{'status':'running','optimizer_steps':0})
        result={'host_gpu_probe':gpu,'acquire_recent_mmmu':images}[a.stage](run);result.update(status='completed',optimizer_steps=0,ended_utc=now())
    except BaseException as e:
        result={'status':'blocked' if isinstance(e,Blocked) else 'error','error':repr(e),'optimizer_steps':0,'ended_utc':now()}
        (run/'logs/error.log').write_text(traceback.format_exc())
    save(run/'metrics.json',result)
    with (run/'experiment.md').open('a') as f:f.write('\n## Outcome\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\nLimitations: current environment probe is not a host health verdict. Failed prerequisites stop dependent work; no original model inference or training started.\n')
    index();print(run);print(json.dumps(result));return result['status']!='completed'

if __name__=='__main__':raise SystemExit(main())
