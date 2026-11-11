"""Final-only outer paper test image materialization; never selects settings."""
import io,json,pathlib,requests,tarfile,shutil,collections
from PIL import Image
from huggingface_hub import hf_hub_url
from data import R,O,D,P,REPOS,META,paper,opts_clean,label,sha,write_rows,save
assert (O/'selection_locked.json').exists(),'Source dev settings/model must be locked first'
if (P/'arxiv_test.jsonl').exists():
 assert (P/'arxiv_test_receipt.json').exists();raise SystemExit('Existing final test materialization preserved')
assert shutil.disk_usage(D).free>7948549045+2*2**30,'Conservative full archive-image storage reserve'
split=json.loads((P/'paper_splits.json').read_text());test=set(split['groups']['test']);ann=D/'ArxivQA/arxivqa.jsonl';groups=collections.defaultdict(list);excluded=collections.Counter()
for line in ann.read_text().splitlines():
 x=json.loads(line)
 if paper(x['image']) not in test:continue
 try:
  opts=opts_clean(x['options']);ans=label(x['label'],opts)
  if not ans:excluded['ambiguous_label']+=1;continue
  groups[pathlib.Path(x['image']).name].append((x,opts,ans))
 except Exception:excluded['invalid_options']+=1
response=requests.get(hf_hub_url(REPOS[0],'images.tgz',repo_type='dataset',revision=META[REPOS[0]]['revision']),stream=True,timeout=120);response.raise_for_status();out=[];seen=set();destination=D/'final_test_images';destination.mkdir(exist_ok=False)
with tarfile.open(fileobj=response.raw,mode='r|gz') as tf:
 for member in tf:
  name=pathlib.Path(member.name).name
  if not member.isfile() or name not in groups:continue
  try:
   im=Image.open(io.BytesIO(tf.extractfile(member).read())).convert('RGB');path=destination/(name+'.png');im.save(path)
  except Exception:excluded['image_decode_failure']+=len(groups[name]);continue
  for x,options,answer in groups[name]:out.append({'id':'ArxivQA:'+str(x['id']),'source':'ArxivQA','question':x['question'],'options':options,'answer':answer,'reason':None,'images':[str(path)],'image_sha256':[sha(path)],'paper_id':paper(x['image']),'split':'test','source_revision':META[REPOS[0]]['revision']})
  seen.add(name)
response.close();assert len({r['id'] for r in out})==len(out);assert all(r['paper_id'] not in set(split['groups']['train'])|set(split['groups']['dev']) for r in out);write_rows(P/'arxiv_test.jsonl',out);save(P/'arxiv_test_receipt.json',{'n':len(out),'images':len(seen),'annotation_sha256':sha(ann),'rows_sha256':sha(P/'arxiv_test.jsonl'),'unavailable_images':len(set(groups)-seen),'exclusions':dict(excluded),'source_config_sha256':sha(O/'config.json'),'selection_locked_sha256':sha(O/'selection_locked.json')})
