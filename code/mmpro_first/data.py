"""Pinned official data, bounded smoke acquisition, paper-disjoint splits, no models."""
import ast,collections,concurrent.futures,hashlib,io,json,os,pathlib,random,re,tarfile,time,zipfile
import requests
from huggingface_hub import hf_hub_download,hf_hub_url,HfApi
from PIL import Image
R=pathlib.Path(os.environ.get('MMPRO_PROJECT','/home/user/mmdl-work'));O=pathlib.Path(os.environ.get('MMPRO_WORK',R/'.restore/mmpro_first_v1'));D=pathlib.Path(os.environ.get('MMPRO_RAW',R/'datasets/raw/MMProFirst_20261111'));P=pathlib.Path(os.environ.get('MMPRO_DATA',R/'datasets/processed/MMProFirst_20261111'));SEED=42
PREP=json.loads((O/'prepare_config.json').read_text()) if (O/'prepare_config.json').exists() else {};TRAIN_CAP=PREP.get('train_cap',16);DEV_CAP=PREP.get('dev_cap',4);SOURCE_CAP=PREP.get('source_cap',16);SCAN=PREP.get('scan_rows',128);PRO_CAP=PREP.get('pro_cap',32)
D.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
def save(p,d):pathlib.Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write_rows(p,rs):pathlib.Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def norm(s):return re.sub(r'\s+',' ',str(s)).strip().casefold()
def paper(path):
 s=pathlib.Path(path).stem
 m=re.search(r'(\d{4}\.\d{4,5})(?:v\d+)?(?:_|$)',s)
 if m:return m[1]
 m=re.match(r'(.+?)_\d',s);return re.sub(r'v\d+$','',m[1]) if m else None
REPOS=['MMInstruction/ArxivQA','OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking','OpenMMReasoner/OpenMMReasoner-SFT-874K','MMMU/MMMU_Pro']
META={r:json.loads((O/(r.split('/')[-1]+'_metadata.json')).read_text()) for r in REPOS}

def rows_api(repo,config,split,n):
 m=META[repo];out=[];obj=None
 if repo in [REPOS[1],REPOS[3]]:
  from pyarrow import parquet as pq
  rel='data/train-00000-of-00018.parquet' if repo==REPOS[1] else config+'/test-00000-of-'+('00004' if config=='vision' else '00002')+'.parquet'
  fp=D/'pinned_shards'/repo.split('/')[-1]/rel
  assert fp.exists(),'Wait for verified direct shard: '+str(fp)
  out=[]
  files=sorted(fp.parent.glob('*.parquet')) if PREP.get('all_pro_shards') and repo==REPOS[3] else [fp]
  for shard in files:
   for batch in pq.ParquetFile(shard).iter_batches(batch_size=min(n,128)):
    out.extend(batch.to_pylist())
    if len(out)>=n:return out[:n]
  return out
 for offset in range(0,n,16):
  for attempt in range(4):
   r=requests.get('https://datasets-server.huggingface.co/rows',params={'dataset':repo,'config':config,'split':split,'offset':offset,'length':min(16,n-offset),'revision':m['revision']},timeout=120)
   if r.ok:break
   time.sleep(1+attempt*2)
  r.raise_for_status();part=r.json();out += [x['row'] for x in part['rows']];obj=part
  if any(x.get('truncated_cells') for x in part['rows']):print('VIEWER_TRUNCATION_REJECT_REQUIRED',repo,offset,flush=True)
 assert HfApi().dataset_info(repo).sha==m['revision'],'Dataset main changed during pinned viewer acquisition'
 save(D/(repo.split('/')[-1]+'_'+re.sub('[^A-Za-z0-9]','_',config)+'_rows.json'),{'rows':out,'revision':m['revision'],'config':config,'range':[0,n]});return out

def image_from_asset(value,repo,key):
 assert isinstance(value,dict)
 if value.get('bytes') is not None:content=value['bytes']
 else:
  url=value['src'];assert META[repo]['revision'] in url,'Viewer image not pinned';b=requests.get(url,timeout=90);b.raise_for_status();content=b.content
 p=D/'images'/(hashlib.sha256((repo+key).encode()).hexdigest()+'.png');p.parent.mkdir(exist_ok=True)
 im=Image.open(io.BytesIO(content)).convert('RGB')
 if not p.exists():im.save(p)
 return str(p)

def opts_clean(options):
 if isinstance(options,str):options=ast.literal_eval(options)
 assert isinstance(options,list) and len(options)>=2
 return [re.sub(r'^\s*\(?[A-Z][).:]\s*','',str(v)).strip() for v in options]
def label(value,opts):
 s=str(value).strip().strip('$').strip();s=re.sub(r'\\boxed\{([^{}]+)\}',r'\1',s);keys=[chr(65+i) for i in range(len(opts))]
 if s.upper() in keys:return s.upper()
 m=re.fullmatch(r'\(?([A-Z])\)?[).:]\s*(.+)',s,re.I)
 if m and m[1].upper() in keys and norm(m[2]).rstrip('.!')==norm(opts[ord(m[1].upper())-65]).rstrip('.!'):return m[1].upper()
 hits=[k for k,v in zip(keys,opts) if norm(v).rstrip('.!')==norm(s).rstrip('.!')];return hits[0] if len(hits)==1 else None

def answer_text(s):
 s=str(s).strip();s=re.sub(r'^\$|\$$','',s).strip();s=s.replace('\\,','').replace(',','');return norm(s).rstrip('.')
def teacher(text,gold,opts=None):
 m=re.search(r'<think>(.*?)</think>',text,re.S)
 if not m:return None
 why=m[1].strip();rest=text[m.end():];a=re.search(r'<answer>(.*?)</answer>',rest,re.S)
 if a:final=a[1].strip()
 else:
  boxes=re.findall(r'\\boxed\{([^{}]+)\}',rest);final=boxes[-1] if boxes else ''
 if not why or not final:return None
 if opts:
  extracted=label(final,opts);return (why,extracted) if extracted and extracted==gold else None
 return (why,str(gold).strip()) if answer_text(final)==answer_text(gold) else None

class Remote(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.bytes=0;self.requests=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset;return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  start=self.pos;end=start+n-1;r=requests.get(self.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=120)
  if r.status_code!=206:r.close();raise RuntimeError('Server must support bounded range: '+str(r.status_code))
  cr=r.headers.get('content-range','');assert cr.startswith(f'bytes {start}-{end}/'),cr
  b=r.content;r.close();assert len(b)==n;self.pos+=n;self.bytes+=n;self.requests+=1;return b

def main():
 assert not (P/'manifest.json').exists(),'New run/version required; preserve frozen manifest'
 import shutil
 assert shutil.disk_usage(D).free>4*2**30
 plan={'available_bytes':shutil.disk_usage(D).free,'ArxivQA_full_images_archive_bytes':7948549045,'strategy':'Complete ArxivQA annotation; bounded tar stream images. Pinned direct Parquet with embedded images for MMFineReason/MMMU-Pro; fixed OMM parquet plus bounded ZIP range images. No full corpus claim.','sample_cap_per_training_source':SOURCE_CAP,'evaluation_cached_per_MMMU_Pro_config':PRO_CAP,'seed':42}
 save(O/'download_plan.json',plan);print('DOWNLOAD_PLAN',plan,flush=True)
 repo=REPOS[0];m=META[repo];path=pathlib.Path(hf_hub_download(repo,'arxivqa.jsonl',repo_type='dataset',revision=m['revision'],local_dir=D/'ArxivQA'));allrows=[json.loads(l) for l in path.read_text().splitlines()];groups=sorted({paper(x['image']) for x in allrows if paper(x['image'])});random.Random(42).shuffle(groups);e=int(len(groups)*.8);outertrain=set(groups[:e]);test=set(groups[e:]);dev=set(groups[:max(1,int(e*.05))]);train=outertrain-dev;assert not train&dev and not outertrain&test
 split={'seed':42,'grouping':'arXiv paper ID extracted from image filename; version removed','annotation_sha256':sha(path),'groups':{'train':sorted(train),'dev':sorted(dev),'test':sorted(test)},'counts':{k:sum(paper(x['image']) in g for x in allrows) for k,g in [('train',train),('dev',dev),('test',test)]},'test_not_used_for_selection_training_or_smoke_evaluation':True};save(P/'paper_splits.json',split)
 byimage=collections.defaultdict(list)
 for x in allrows:byimage[pathlib.Path(x['image']).name].append(x)
 if (P/'arxiv_train.jsonl').exists() and (P/'arxiv_dev.jsonl').exists():
  acquired={k:[json.loads(l) for l in (P/('arxiv_'+k+'.jsonl')).read_text().splitlines()] for k in ['train','dev']};reject=collections.Counter()
 else:
  acquired={'train':[],'dev':[]};seen=set();reject=collections.Counter();url=hf_hub_url(repo,'images.tgz',repo_type='dataset',revision=m['revision']);res=requests.get(url,stream=True,timeout=120);res.raise_for_status();start=time.time();stream=res.raw;stream.decode_content=False
  with tarfile.open(fileobj=stream,mode='r|gz') as tf:
   for member in tf:
    if not member.isfile() or pathlib.Path(member.name).name not in byimage:continue
    name=pathlib.Path(member.name).name;pid=paper(name);kind='train' if pid in train else 'dev' if pid in dev else 'test'
    if kind=='test' or len(acquired[kind])>=(TRAIN_CAP if kind=='train' else DEV_CAP) or pid in seen:continue
    valid=[]
    for x in byimage[name]:
     try:
      opts=opts_clean(x['options']);lab=label(x['label'],opts)
      if not lab or not x.get('rationale','').strip():reject['arxiv_invalid_label_or_rationale']+=1;continue
      valid.append((x,opts,lab))
     except Exception:reject['arxiv_bad_options']+=1
    if not valid:continue
    b=tf.extractfile(member).read();im=Image.open(io.BytesIO(b)).convert('RGB');dest=D/'images'/('arxiv_'+name+'.png');dest.parent.mkdir(exist_ok=True);im.save(dest);x,opts,lab=valid[0];row={'id':'ArxivQA:'+str(x['id']),'source':'ArxivQA','question':x['question'],'options':opts,'answer':lab,'reason':x['rationale'].strip(),'images':[str(dest)],'paper_id':pid,'source_id':str(x['id']),'source_revision':m['revision'],'original_image_path':x['image'],'split':kind};acquired[kind].append(row);seen.add(pid);print('ARXIV_IMAGE',kind,row['id'],len(acquired[kind]),flush=True)
    if len(acquired['train'])==TRAIN_CAP and len(acquired['dev'])==DEV_CAP:break
  res.close();print('ARXIV_STREAM_COMPLETE',len(allrows),split['counts'],time.time()-start,flush=True)
  write_rows(P/'arxiv_train.jsonl',acquired['train']);write_rows(P/'arxiv_dev.jsonl',acquired['dev'])
 # Bounded evaluation cache only. Gold/explanation remain grader-side, never input builders.
 evaluation=[]
 for config in ['standard (10 options)','vision']:
  raw=rows_api(REPOS[3],config,'test',PRO_CAP);normalized=[]
  for x in raw:
   if config=='vision':ims=[image_from_asset(x['image'],REPOS[3],config+x['id'])];q='Read and solve the multiple-choice question in the screenshot. Follow the option labels shown in the screenshot.'
   else:
    ims=[image_from_asset(x[f'image_{i}'],REPOS[3],config+x['id']+str(i)) for i in range(1,8) if x.get(f'image_{i}')];q=x['question']
   opts=opts_clean(x['options']);assert x['answer'] in [chr(65+i) for i in range(len(opts))];normalized.append({'id':x['id'],'source':'MMMU_Pro_'+('vision' if config=='vision' else 'standard10'),'question':q,'options':opts,'answer':x['answer'],'images':ims,'reason':None,'source_revision':META[REPOS[3]]['revision'],'split':'test','screenshot_only':config=='vision'})
  evaluation+=normalized;print('MMMU_PRO_CACHE',config,len(normalized),flush=True)
 excluded_text={norm(x['question']) for x in allrows if paper(x['image']) in dev|test};excluded_text.update(norm(x['question']) for x in evaluation);excluded_images={sha(p) for x in acquired['dev']+evaluation for p in x['images']};protected_papers=dev|test
 def dedup(row):
  if 'mmmu' in row.get('original_source','').casefold():reject['distill_MMMU_source_excluded']+=1;return False
  if norm(row['question']) in excluded_text:reject['protected_exact_question_overlap']+=1;return False
  if any(sha(p) in excluded_images for p in row['images']):reject['protected_exact_image_overlap']+=1;return False
  if any(paper(p) in protected_papers for p in row.get('source_image_paths',[])):reject['protected_arxiv_paper_overlap']+=1;return False
  return True
 # Apply known protected-image/text checks to ArxivQA itself before any training.
 acquired['train']=[r for r in acquired['train'] if dedup(r)]
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/dev/shm/mmdl-user/ebb281ec70b05090aa6165b016eac8ec08e71b17',local_files_only=True)
 fine=[];fr=rows_api(REPOS[1],'default','train',SCAN)
 for x in fr:
  if len(fine)>=SOURCE_CAP:break
  if not x.get('is_consistent'):reject['fine_inconsistent']+=1;continue
  target=teacher(x['qwen3vl_235b_thinking_response'],x['answer'])
  if target is None:reject['fine_teacher_final_not_exact_annotation']+=1;continue
  why,ans=target
  if len(tokenizer.encode(why,add_special_tokens=False))>1800:reject['fine_reason_too_long_no_truncation']+=1;continue
  im=image_from_asset(x['image'],REPOS[1],str(x['id']));row={'id':'MMFineReason:'+str(x['id']),'source':'MMFineReason','question':x['question'].replace('<image>','').strip(),'options':[],'answer':ans,'reason':why,'images':[im],'source_id':str(x['id']),'original_source':x['source'],'source_revision':META[REPOS[1]]['revision'],'teacher_field':'qwen3vl_235b_thinking_response','pass_rate':x.get('pass_rate'),'consistency_flag':x['is_consistent'],'teacher_final_matches_annotation_exact':True,'split':'train'}
  if dedup(row):fine.append(row)
 print('FINE_VALID',len(fine),dict(reject),flush=True)
 # All five source config schemas are inspected; selected data file is only the 6.6MB config.
 configs=['llava_cot','OpenVLThinker-sft-iter3','WeMath','m1_sft','mmr1'];config_schema={}
 for c in configs:
  sample=rows_api(REPOS[2],c,'train',1)[0];config_schema[c]={'fields':list(sample),'messages_roles':[v['role'] for v in sample['messages']],'image_urls':[v['image_url']['url'] for msg in sample['messages'] for v in msg['content'] if v.get('image_url')]}
 save(O/'OpenMMReasoner_configs.json',config_schema)
 from pyarrow import parquet as pq
 cp=pathlib.Path(hf_hub_download(REPOS[2],'parquet/OpenVLThinker-sft-iter3.parquet',repo_type='dataset',revision=META[REPOS[2]]['revision'],local_dir=D/'OpenMMReasoner'));omrows=pq.read_table(cp).to_pylist()[:SCAN];archives=[];openmm=[]
 for idx in [1,2]:
  filename=f'image_zip/images_{idx:03d}.zip';info=next(f for f in META[REPOS[2]]['files'] if f['name']==filename);remote=Remote(hf_hub_url(REPOS[2],filename,repo_type='dataset',revision=META[REPOS[2]]['revision']),info['bytes']);zf=zipfile.ZipFile(remote);names=zf.namelist();archives.append((remote,zf,names));print('ZIP_INDEX',idx,len(names),remote.bytes,flush=True)
 for x in omrows:
  if len(openmm)>=SOURCE_CAP:break
  messages=x['messages'];uq='\n'.join(v['text'] for msg in messages if msg['role']=='user' for v in msg['content'] if v.get('text'));response='\n'.join(v['text'] for msg in messages if msg['role']=='assistant' for v in msg['content'] if v.get('text'));uq=uq.replace('\\n','\n');parts=list(re.finditer(r'(?m)^\(?([A-J])[).:]\s*(.*)',uq));opts=[p[2].strip() for p in parts];q=uq[:parts[0].start()].rstrip().removesuffix('Choices:').strip() if parts else uq;ansmatch=re.search(r'<answer>(.*?)</answer>',response,re.S)
  if not ansmatch:reject['openmm_no_answer_tag']+=1;continue
  ans=label(ansmatch[1],opts) if opts else ansmatch[1].strip();rr=teacher(response,ans,opts) if ans else None
  if not rr:reject['openmm_bad_target']+=1;continue
  why,ans=rr
  if len(tokenizer.encode(why,add_special_tokens=False))>1800:reject['openmm_reason_too_long_no_truncation']+=1;continue
  urls=[v['image_url']['url'] for msg in messages if msg['role']=='user' for v in msg['content'] if v.get('image_url')];paths=[]
  try:
   for path in urls:
    found=None
    for remote,zf,names in archives:
     matches=[n for n in names if n.endswith(path.lstrip('/')) or n.endswith('/'+pathlib.Path(path).name)]
     if len(matches)==1:found=(zf,matches[0]);break
    if found is None:raise ValueError('image archive member missing/ambiguous')
    b=found[0].read(found[1]);im=Image.open(io.BytesIO(b)).convert('RGB');dest=D/'images'/('openmm_'+hashlib.sha256(path.encode()).hexdigest()+'.png');im.save(dest);paths.append(str(dest))
  except Exception as e:reject['openmm_image_failure']+=1;continue
  row={'id':'OpenMMReasoner:'+str(x['id']),'source':'OpenMMReasoner','question':q,'options':opts,'answer':ans,'reason':why,'images':paths,'source_image_paths':urls,'original_source':'OpenVLThinker-sft-iter3','source_revision':META[REPOS[2]]['revision'],'split':'train','answer_validation':'provided_assistant_answer_parsed_and_mapped_to_actual_options; no_independent_GT_field_in_this_config'}
  if paths and dedup(row):openmm.append(row)
 print('OPENMM_VALID',len(openmm),dict(reject),flush=True)
 for r in acquired['train']+acquired['dev']+fine+openmm+evaluation:
  r['image_sha256']=[sha(p) for p in r['images']]
  for p in r['images']:
   with Image.open(p) as im:im.verify()
 for name,rs in [('arxiv_train',acquired['train']),('arxiv_dev',acquired['dev']),('fine_train',fine),('openmm_train',openmm),('mmmu_pro_cached',evaluation)]:write_rows(P/(name+'.jsonl'),rs)
 manifest={'seed':42,'model_revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17','source_revisions':{r:META[r]['revision'] for r in REPOS},'source_counts':{'ArxivQA':len(acquired['train']),'MMFineReason':len(fine),'OpenMMReasoner':len(openmm),'arxiv_dev':len(acquired['dev']),'MMMU_Pro_cached':len(evaluation)},'files':{p.name:sha(p) for p in P.glob('*.jsonl')},'paper_splits_sha256':sha(P/'paper_splits.json'),'exclusions':dict(reject),'OpenMMReasoner_selected_config':'OpenVLThinker-sft-iter3','OpenMMReasoner_parquet_sha256':sha(cp),'archive_range_transfers':[{'bytes':r.bytes,'requests':r.requests,'members':len(n)} for r,z,n in archives],'dedup_scope':{'ArxivQA_dev_test_questions':'all annotations','ArxivQA_dev_test_image_hashes':'acquired dev images only; test images not fully downloaded','MMMU_Pro':f'cached first{PRO_CAP} per config, exact text and decoded PNG SHA256','source_arxiv_paper_ids':'all protected dev/test groups when image path provides paper ID','limitations':'No perceptual duplicate search; no claim of full-corpus image overlap exclusion. OMM selected config lacks independently separate gold.'},'test_evaluation_executed':False,'ready':bool(acquired['train'] and acquired['dev'] and fine and openmm)};save(P/'manifest.json',manifest);print('DATA_READY',json.dumps(manifest),flush=True)
if __name__=='__main__':main()
