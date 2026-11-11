"""Create a distinct run; train/eval commands never mutate the first smoke run."""
import argparse,hashlib,json,os,pathlib,random,shutil,subprocess,sys,time,concurrent.futures
from huggingface_hub import HfApi,hf_hub_download
R=pathlib.Path('/home/user/mmdl-work');CODE=pathlib.Path(__file__).parent;PY=sys.executable
REVISIONS={'MMInstruction/ArxivQA':'85a6dca0e2bdc6f0268ae519be8913f83a83cafd','OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking':'cda6140bfba449bef9e57065a528b7424966e144','OpenMMReasoner/OpenMMReasoner-SFT-874K':'c44cf46e41bd24351ae7e7771a44fcd2dbf75f0a','MMMU/MMMU_Pro':'0d7426df4ccb3d8704a992fe6850abee7c262127'}
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','final']);ap.add_argument('--run-id',required=True);ap.add_argument('--profile',choices=['smoke','expanded'],default='smoke');ap.add_argument('--model-path',default='/dev/shm/mmdl-user/ebb281ec70b05090aa6165b016eac8ec08e71b17');a=ap.parse_args();assert a.run_id.replace('_','').replace('-','').isalnum();work=R/'.restore'/a.run_id;raw=R/'datasets/raw'/a.run_id;data=R/'datasets/processed'/a.run_id;art=R/'runs'/a.run_id
 env={**os.environ,'MMPRO_WORK':str(work),'MMPRO_RAW':str(raw),'MMPRO_DATA':str(data),'MMPRO_ART':str(art),'HF_HUB_DISABLE_XET':'1'}
 def stage(name,model=None,extra=()):
  key=name+('_'+model if model else '')+('_'+extra[list(extra).index('--evaluation-name')+1] if '--evaluation-name' in extra else '')
  if (art/key/'complete.json').exists():
   assert json.loads((art/key/'config.json').read_text())==json.loads((work/'config.json').read_text()), 'Configuration changed; completed stage cannot be reused'
   assert json.loads((art/key/'complete.json').read_text())['manifest_sha256']==sha(data/'manifest.json'), 'Data changed; completed stage cannot be reused'
   print('Completed; preserved:',key,flush=True);return
  cmd=[PY,'-u',str(CODE/'run.py'),name]+(['--model',model] if model else [])+list(extra)
  with (work/(key+'.log')).open('a') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 if a.action=='prepare':
  assert not work.exists(),'Existing run must be preserved; choose a new run ID'
  for p in [work,raw,data,art]:p.mkdir(parents=True)
  sys.path.insert(0,str(CODE));from core import DEFAULT
  cfg={**DEFAULT,'model_path':a.model_path,'mining_questions':4 if a.profile=='smoke' else 32}
  if a.profile=='expanded':cfg.update(sft_updates=128,preference_updates=16)
  dump(work/'config.json',cfg);dump(work/'prepare_config.json',{} if a.profile=='smoke' else {'train_cap':128,'dev_cap':32,'source_cap':128,'scan_rows':2048,'pro_cap':1730,'all_pro_shards':True})
  dump(work/'profile_scope.json',{'profile':a.profile,'expanded_is_bounded_pool_not_complete_training_corpus':True,'source_cap':16 if a.profile=='smoke' else 128,'settings_fixed_before_training':True})
  needed=[]
  for repo,rev in REVISIONS.items():
   info=HfApi().dataset_info(repo,revision=rev,files_metadata=True);files=[{'name':f.rfilename,'bytes':f.size} for f in info.siblings];dump(work/(repo.split('/')[-1]+'_metadata.json'),{'revision':rev,'files':files})
   for f in files:
    if repo.endswith('MMMU_Pro') and f['name'].startswith(('standard (10 options)/','vision/')) and f['name'].endswith('.parquet') and (a.profile=='expanded' or '00000' in f['name']):needed.append((repo,rev,f))
    if 'MMFineReason' in repo and f['name']=='data/train-00000-of-00018.parquet':needed.append((repo,rev,f))
  # Direct embedded images, not expiring dataset viewer assets. Model weights never downloaded here.
  required=sum(f['bytes'] for _,_,f in needed)+8*2**30;assert shutil.disk_usage(raw).free>required,'Insufficient download + reserve space'
  dump(work/'download_plan.json',{'direct_shard_bytes':sum(f['bytes'] for _,_,f in needed),'image_stream_archive_full_bytes':7948549045,'free_bytes':shutil.disk_usage(raw).free})
  def fetch(item):
   repo,rev,f=item;p=pathlib.Path(hf_hub_download(repo,f['name'],repo_type='dataset',revision=rev,local_dir=raw/'pinned_shards'/repo.split('/')[-1]));assert p.stat().st_size==f['bytes'];return {'repo':repo,'revision':rev,'file':f['name'],'sha256':sha(p),'bytes':f['bytes']}
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:dump(work/'direct_shards_verified.json',list(ex.map(fetch,needed)))
  # Preserve the exact official parser source revision used for smoke scoring.
  import requests
  url='https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/268471d0d488258990025331c7528359c324aa25/mmmu-pro/evaluate.py';response=requests.get(url,timeout=60)
  # Repository path is resolved from the pinned Git tree rather than guessed.
  if response.status_code!=200:
   tree=requests.get('https://api.github.com/repos/MMMU-Benchmark/MMMU/git/trees/268471d0d488258990025331c7528359c324aa25',params={'recursive':'1'},timeout=60);tree.raise_for_status();matches=[v['path'] for v in tree.json()['tree'] if v['path'].endswith('/evaluate.py') and 'pro' in v['path'].lower()];assert len(matches)==1,matches;response=requests.get('https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/268471d0d488258990025331c7528359c324aa25/'+matches[0],timeout=60)
  response.raise_for_status();(work/'evaluate.py').write_text(response.text)
  subprocess.run([PY,'-u',str(CODE/'data.py')],env=env,check=True);subprocess.run([PY,str(CODE/'run.py'),'validate'],env=env,check=True)
  model=pathlib.Path(a.model_path);receipt=json.loads((model/'processor_manifest.json').read_text());assert receipt['revision']==cfg['model_revision'] and receipt['complete']
  for f in receipt['files']:assert sha(model/f['name'])==f['sha256'],f['name']
  dump(work/'model_verified.json',receipt)
  rr=[json.loads(l) for l in (data/'mmmu_pro_cached.jsonl').read_text().splitlines()];sample=[]
  for source in sorted({r['source'] for r in rr}):
   group=[r for r in rr if r['source']==source];random.Random(42).shuffle(group);sample+=group[:min(2 if a.profile=='smoke' else 32,len(group))]
  (work/'evaluation_selection.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sample));return
 assert (data/'manifest.json').exists() and (work/'model_verified.json').exists()
 if a.action=='train':
  for name in ['preflight','arxiv_sft','mixed_sft','mine','dpo','ranking']:stage(name)
  models=['initial','arxiv_sft','mixed_sft','dpo','ranking']
  for model in models:stage('evaluate',model)
  stats={m:json.loads((art/('evaluate_'+m)/'metrics.json').read_text())['by_source']['ArxivQA'] for m in models};selected=max(models,key=lambda m:(stats[m]['free_correct'],stats[m]['ll_correct'],stats[m]['format_ok'],-models.index(m)))
  dump(work/'selection_locked.json',{'selected_model':selected,'dev':stats,'config_sha256':sha(work/'config.json'),'rule':'dev free, LL, format, fixed model order; no MMMU-Pro selection'});stage('pro',selected)
 else:
  lock=json.loads((work/'selection_locked.json').read_text());assert lock['config_sha256']==sha(work/'config.json'),'Config changed after source dev selection'
  assert len([json.loads(l) for l in (data/'mmmu_pro_cached.jsonl').read_text().splitlines()])==3460,'Full Pro cache required; use expanded prepare profile'
  subprocess.run([PY,str(CODE/'materialize_test.py')],env=env,check=True)
  selected=lock['selected_model'];stage('evaluate',selected,['--evaluation-file',str(data/'arxiv_test.jsonl'),'--evaluation-name','locked_arxiv_test']);stage('pro',selected,['--evaluation-file',str(data/'mmmu_pro_cached.jsonl'),'--evaluation-name','locked_full'])
if __name__=='__main__':main()
