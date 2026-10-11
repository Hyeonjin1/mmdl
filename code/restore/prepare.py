"""New seed3407 manifests from pinned originals; not claimed identical to old IDs."""
import argparse, ast, collections, hashlib, json, random, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/p04'),str(ROOT/'code/p03')]
from records import sha,rows,Blocked
from data import image_info,textkey
from acquire import CHART,TAB,PGPS
from scoring_latest_v2 import normalize
CFG=ROOT/'configs/restore/E123.json'

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def stratum(row):
    text=row['question'].lower()
    task=next((name for name,pattern in [('percent','percent'),('ratio','ratio|times'),('difference','difference|more|less|fewer'),('aggregate','sum|total|average|mean|median|combined'),('comparison','most|least|higher|lower|greater')] if re.search(pattern,text)),'other')
    answer=row['answer'];form='percent' if '%' in answer else 'decimal' if '.' in answer else 'integer' if re.fullmatch(r'[+-]?[\d,]+',answer) else 'other'
    return task+':'+form
def half_stratified(data):
    groups=collections.defaultdict(list)
    for row in data:groups[stratum(row)].append(row)
    rng=random.Random(3407);selected=[];remainders=[]
    for key,items in sorted(groups.items()):
        rng.shuffle(items);n=len(items)//2;selected.extend(items[:n]);remainders.append((len(items)%2,key,items[n:]))
    for _,key,items in sorted(remainders,reverse=True):
        if len(selected)==500:break
        if items:selected.append(items[0])
    assert len(selected)==500
    return selected

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['e12','e3']);a=parser.parse_args()
    cfg=json.loads(CFG.read_text());target=Path(cfg['processed_dir']);target.mkdir(parents=True,exist_ok=True)
    if (target/(('E1' if a.mode=='e12' else 'E3')+'.jsonl')).exists():raise Blocked('Already prepared; verify and reuse existing version')
    evaluation=rows(cfg['mmmu_manifest']);eval_questions={textkey(r['question']) for r in evaluation};eval_pixels=set();eval_hashes=[]
    for row in evaluation:
        for name in row['image_path']:
            info=image_info(Path(cfg['mmmu_manifest']).parent/name);eval_pixels.add(info['pixel_sha256']);eval_hashes.append((row['index'],info['dhash']))
    candidates=json.loads((ROOT/'configs/P04/candidates.json').read_text()) if a.mode=='e12' else {}
    rejected=[];cache={};valid={};near=[]
    if a.mode=='e3':
        found=list((ROOT/'datasets/raw/PGPS9K_P04/extracted').rglob('PGPS9K/train.json'))
        if len(found)!=1:raise Blocked('Official PGPS train annotation missing')
        path=found[0];base=path.parent.parent;raw=json.loads(path.read_text());items=[]
        # Exclude duplicate diagrams from official PGPS and Geometry3K tests.
        official_test_pixels=set()
        for testpath in [path.with_name('test.json'),base/'Geometry3K/test.json']:
            if not testpath.exists():continue
            testdata=json.loads(testpath.read_text())
            for item in testdata.values():
                image=base/'Diagram'/item['diagram']
                if image.exists():official_test_pixels.add(image_info(image)['pixel_sha256'])
        if not isinstance(raw,dict):raise Blocked('Unexpected PGPS train schema; inspect before converting')
        for key,item in raw.items():
            try:
                if not isinstance(item.get('text'),str) or not item['text'].strip():raise ValueError('Missing source question')
                if re.search(r'\bN_?\d+\b|\[NUM\]|<num>',item['text']):raise ValueError('Unresolved number placeholders')
                opts=[str(x).strip() for x in item['choices']]
                if not 2<=len(opts)<=26 or len(set(opts))!=len(opts):raise ValueError('Invalid or duplicate options')
                answer=str(item['answer']).strip();matches=[i for i,v in enumerate(opts) if normalize(v)==normalize(answer)]
                if len(matches)!=1:raise ValueError('Source answer does not uniquely match option value; do not infer index')
                # Relations and solution programs never enter model input.
                question=item['text']+'\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(opts))
                items.append(dict(id='PGPS9K:'+str(key),source='PGPS9K',split='train',original_key=str(key),
                   question=question,source_question=item['text'],answer=chr(65+matches[0]),source_answer=answer,options=opts,
                   image=str(base/'Diagram'/item['diagram']),diagram_name=item['diagram'],annotation=str(path),
                   relationship_annotations_present=all(k in item for k in ['parsing_stru_seqs','parsing_sem_seqs']),
                   solution_program_used_as_input=False))
            except Exception as e:rejected.append(dict(id='PGPS9K:'+str(key),reason=repr(e)))
        candidates={'PGPS9K_train':items}
    for name,items in candidates.items():
        valid[name]=[]
        for row in items:
            try:
                if row['image'] not in cache:cache[row['image']]=image_info(row['image'])
                info=cache[row['image']];r={**row,**info}
                if a.mode=='e3' and r['pixel_sha256'] in official_test_pixels:raise ValueError('Diagram overlaps official PGPS/Geometry3K test; excluded')
                if not r['question'].strip() or not r['answer'].strip():raise ValueError('Empty question/answer')
                if r['source']=='ChartQA':
                    original=json.loads((ROOT/'datasets/raw/ChartQA_P04'/(r['split']+'_human.json')).read_text())[r['original_index']]
                    assert r['question']==original['query'] and r['answer']==str(original['label']).strip() and r['chart_name']==original['imgname']
                    r['table_group']=sha(r['review_table'])
                elif r['source']=='TabMWP':
                    original=json.loads((ROOT/'datasets/raw/TabMWP_P04'/r['review_annotation']).read_text())[r['original_key']]
                    assert r['question']==original['question'] and r['answer']==str(original['answer'])
                    r['table_group']=hashlib.sha256(textkey(r['table_text']).encode()).hexdigest()
                    # No rationale or table text in the processed user question.
                    r.pop('solution',None);r.pop('table_text',None)
                else:r['table_group']=r['pixel_sha256']
                if textkey(r.get('source_question',r['question'])) in eval_questions or r['pixel_sha256'] in eval_pixels:raise ValueError('Exact MMMU text/RGB overlap')
                close=[qid for qid,h in eval_hashes if (r['dhash']^h).bit_count()<=2]
                if close:near.append({'id':r['id'],'MMMU_candidates':close});raise ValueError('Near MMMU image excluded pending review')
                valid[name].append(r)
            except Exception as e:rejected.append({'id':row['id'],'reason':repr(e)})
    # Connected components of same decoded image or same source table across splits.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for items in valid.values():
        for r in items:
            x='pixel:'+r['pixel_sha256'];y=r['source']+':table:'+r['table_group'];parent[find(y)]=find(x)
    for items in valid.values():
        for r in items:r['group_id']=find('pixel:'+r['pixel_sha256'])
    def unique(items):
        seen=set();out=[]
        for r in items:
            key=(r['pixel_sha256'],textkey(r['question']))
            if key not in seen:seen.add(key);out.append(r)
        return out
    rng=random.Random(3407);selected={};heldgroups=set()
    sources=['ChartQA','TabMWP'] if a.mode=='e12' else ['PGPS9K']
    for source in sources:
        key={'ChartQA':'ChartQA_val','TabMWP':'TabMWP_dev','PGPS9K':'PGPS9K_train'}[source]
        pool=unique(valid[key]);rng.shuffle(pool);selected[source+'_check']=pool[:100]
        if len(pool)<100:raise Blocked('Not enough independent check rows:'+source)
        heldgroups.update(r['group_id'] for r in pool[:100])
    for source in sources:
        pool=unique([r for r in valid[source+'_train'] if r['group_id'] not in heldgroups]);rng.shuffle(pool)
        n=500 if source=='TabMWP' else 1000
        if len(pool)<n:
            write(target/('insufficient_'+a.mode+'.json'),{'available':len(pool),'required':n,'source':source,'rejections':rejected});raise Blocked('Not enough safe unique training rows:'+source)
        selected[source+'_train']=pool[:n]
    if a.mode=='e12':selected.update(E1=selected['ChartQA_train'],E2=half_stratified(selected['ChartQA_train'])+selected['TabMWP_train'])
    else:selected['E3']=selected['PGPS9K_train']
    files={}
    for name,data in selected.items():
        p=target/(name+'.jsonl');p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data));files[name]={'path':str(p),'sha256':sha(p),'count':len(data)}
    ready=target/'manifest.json'
    old=json.loads(ready.read_text()) if ready.exists() else {}
    files={**old.get('files',{}),**files}
    b0=Path(cfg['b0_run'])/'evaluation_protocol.json'
    manifest={'processing_version':'RESTORE_v1_new_selection_seed3407','previous_sample_identity_verified':False,
       'identity_note':'Published repository did not include previous processed manifests/QA IDs; regenerated new seed3407 version',
       'seed':3407,'ready_for_training':all(n in files for n in ['E1','E2','ChartQA_check','TabMWP_check']),
       'E3_ready': 'E3' in files,'files':files,'sources':{'ChartQA':CHART,'TabMWP':TAB,'PGPS9K':PGPS},
       'B0_protocol_sha256':sha(b0),'MMMU_manifest_sha256':sha(cfg['mmmu_manifest']),
       'MMMU_image_overlap_checked':True,'train_check_groups_disjoint':True,'semantic_duplicate_check_exhaustive':False,
       'stratification_note':'E2 ChartQA half by new question-operation and answer-form heuristics; not historical MMMU type labels'}
    ready.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    write(target/('preparation_audit_'+a.mode+'.json'),{'rejected':rejected,'near_MMMU_excluded':near,
          'counts':{n:len(r) for n,r in selected.items()},'available_candidates':{n:len(r) for n,r in valid.items()},
          'distributions':{n:dict(collections.Counter(stratum(r) for r in data)) for n,data in selected.items()}})
    if a.mode=='e3':cfg['check_datasets']=['ChartQA','TabMWP','PGPS9K'];CFG.write_text(json.dumps(cfg,indent=2)+'\n')
    print('PREPARED',a.mode,{n:len(r) for n,r in selected.items()},flush=True)
if __name__=='__main__':main()
