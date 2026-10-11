"""Decode images and preserve source answers; never infer missing annotations."""
import ast,csv,hashlib,json,random,re,sys
from collections import Counter
from pathlib import Path
from PIL import Image
from records import ROOT,Blocked,save,sha,rows,jsonl
from acquire import RAW,CHART,TAB,PGPS
from scoring import normalize

def textkey(s):return re.sub(r'\s+',' ',re.sub(r'<image\s+\d+>','',str(s)).casefold()).strip()
def image_info(path):
    with Image.open(path) as image:
        image.load();im=image.convert('RGB')
        pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        small=list(im.convert('L').resize((9,8)).getdata());dhash=sum((small[y*9+x]>small[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
        return dict(image_sha256=sha(path),pixel_sha256=pixel,image_size=list(im.size),dhash=dhash)

def mmmu(cfg,run):
    from datasets import DatasetDict,load_dataset,load_from_disk
    sys.path.insert(0,str(ROOT/'code'))
    from eval_mmmu_v3 import SUBJECTS,DATA_REVISION,select_examples
    cache=ROOT/'data/mmmu_validation'
    ds=load_from_disk(str(cache)) if cache.exists() else DatasetDict({s:load_dataset('MMMU/MMMU',s,split='validation',revision=DATA_REVISION) for s in SUBJECTS})
    selected=select_examples(ds,0);target=RAW/'MMMU_P04';target.mkdir(exist_ok=True)
    records=[]
    for subject,i in selected:
        item=ds[subject][i];opts=item['options'];opts=ast.literal_eval(opts) if isinstance(opts,str) else opts
        record=dict(id=item['id'],subject=subject,source='MMMU',split='validation',question=item['question'],answer=item['answer'],question_type=item['question_type'],options=opts if item['question_type']=='multiple-choice' else [],images={},image_metadata={})
        for k in range(1,8):
            image=item.get(f'image_{k}')
            if image is None:continue
            path=target/'images'/(item['id']+f'_{k}.png');path.parent.mkdir(exist_ok=True)
            if not path.exists():image.convert('RGB').save(path)
            record['images'][f'image_{k}']=str(path);record['image_metadata'][f'image_{k}']=image_info(path)
        records.append(record)
    assert len(records)==900 and Counter(r['subject'] for r in records)=={s:30 for s in SUBJECTS}
    jsonl(target/'validation.jsonl',records)
    manifest=dict(dataset='MMMU/MMMU',revision=DATA_REVISION,split='validation',count=900,subjects=dict(Counter(r['subject'] for r in records)),evaluation_only=True,file_sha256=sha(target/'validation.jsonl'))
    save(target/'manifest.json',manifest);save(run/'data_manifest.json',manifest)
    return dict(downloaded_questions=900,subject_count=30)

def pgps_rows():
    found=list((RAW/'PGPS9K_P04/extracted').rglob('PGPS9K/train.json'))
    if len(found)!=1:raise Blocked('PGPS9K official archive missing or unexpected layout: expected PGPS9K/train.json')
    path=found[0];base=path.parent.parent
    data=json.loads(path.read_text());result=[]
    for key,item in data.items():
        for field in ['text','diagram','answer','choices','expression','parsing_stru_seqs','parsing_sem_seqs']:
            if field not in item:raise ValueError('PGPS schema missing '+field)
        if not isinstance(item['text'],str) or re.search(r'\bN_?\d+\b|\[NUM\]|<num>',item['text']):continue
        choices=[str(x) for x in item['choices']];answer=str(item['answer'])
        match=[i for i,x in enumerate(choices) if normalize(x)==normalize(answer)]
        if len(match)!=1:continue # no guessing whether an ambiguous numeric answer is an index
        result.append(dict(id='PGPS9K:'+str(key),source='PGPS9K',split='train',question=item['text'],answer=chr(65+match[0]),source_answer=answer,options=choices,image=str(base/'Diagram'/item['diagram']),diagram_name=item['diagram'],review_annotation=str(path),original_key=str(key),solution_program=item['expression']))
    geometry=set()
    for p in base.glob('Geometry3K/*.json'):
        obj=json.loads(p.read_text())
        if isinstance(obj,dict):
            for item in obj.values():
                if isinstance(item,dict) and 'diagram' in item:
                    image=base/'Diagram'/item['diagram']
                    if image.is_file():geometry.add(image_info(image)['pixel_sha256'])
    return result,geometry

def prepare(cfg,run):
    candidate=json.loads((ROOT/'configs/P04/candidates.json').read_text())
    rejected=[];valid={};info_cache={}
    for key,items in candidate.items():
        valid[key]=[]
        for row in items:
            try:
                info=info_cache.setdefault(row['image'],image_info(row['image']))
                row={**row,**info};assert row['question'].strip() and row['answer'].strip()
                if re.search(r'\[NUM\]|<num>|\bN_\d+\b',row['question']):raise ValueError('Unresolved numeric placeholder')
                if row['source']=='ChartQA':
                    table=Path(row['review_table']);table_rows=list(csv.reader(table.open()))
                    if len(table_rows)<2:raise ValueError('Review CSV empty')
                    row['review_table_sha256']=sha(table);row['table_group']=sha(table)
                else:row['table_group']=hashlib.sha256(textkey(row['table_text']).encode()).hexdigest()
                valid[key].append(row)
            except Exception as e:rejected.append(dict(id=row['id'],reason=repr(e)))
    pgps_error=None;geometry=set()
    try:
        pgps,geometry=pgps_rows();valid['PGPS9K_train']=[]
        for row in pgps:
            try:valid['PGPS9K_train'].append({**row,**image_info(row['image'])})
            except Exception as e:rejected.append(dict(id=row['id'],reason=repr(e)))
    except Blocked as e:pgps_error=str(e)
    mmmu_path=RAW/'MMMU_P04/validation.jsonl'
    evaluation=rows(mmmu_path) if mmmu_path.exists() else []
    questions={textkey(r['question']) for r in evaluation};images={m['pixel_sha256'] for r in evaluation for m in r['image_metadata'].values()}
    if not evaluation:
        saved=ROOT/'results/full900_v3/predictions.jsonl'
        if saved.exists():
            for r in rows(saved):
                prompt=r['prompt'].split('\n\nChoices:\n')[0]
                prompt=re.sub(r'\n\nSolve the problem\..*','',prompt,flags=re.S)
                questions.add(textkey(prompt))
    eval_hashes=[(r['id'],m['dhash']) for r in evaluation for m in r['image_metadata'].values()]
    suspects=[]
    for key,items in list(valid.items()):
        good=[]
        for row in items:
            duplicate=textkey(row['question']) in questions or row['pixel_sha256'] in images
            near=[qid for qid,d in eval_hashes if (row['dhash']^d).bit_count()<=2]
            if duplicate:rejected.append(dict(id=row['id'],reason='confirmed exact MMMU question or decoded image overlap'));continue
            if near:
                suspects.append(dict(id=row['id'],candidate_mmmu_ids=near,status='unreviewed_perceptual_candidate_excluded'))
                continue
            good.append(row)
        valid[key]=good
    # Union all identical images or source tables across official splits before splitting.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def group(row):
        a='pixel:'+row['pixel_sha256'];b=row['source']+':table:'+row.get('table_group',row['pixel_sha256'])
        pa,pb=find(a),find(b);parent[pb]=pa;return a
    for items in valid.values():
        for row in items:group(row)
    for items in valid.values():
        for row in items:row['group_id']=find('pixel:'+row['pixel_sha256'])
    rng=random.Random(3407);selected={};held_groups=set()
    def unique(items):
        seen=set();out=[]
        for row in items:
            key=(row['pixel_sha256'],textkey(row['question']))
            if key not in seen:seen.add(key);out.append(row)
        return out
    for source,val_key in [('ChartQA','ChartQA_val'),('TabMWP','TabMWP_dev'),('PGPS9K','PGPS9K_train')]:
        pool=unique(valid.get(val_key,[]));rng.shuffle(pool)
        chosen=pool[:100];selected[source+'_check']=chosen;held_groups.update(r['group_id'] for r in chosen)
    for source,n in [('ChartQA',1000),('TabMWP',500),('PGPS9K',1000)]:
        pool=unique([r for r in valid.get(source+'_train',[]) if r['group_id'] not in held_groups]);rng.shuffle(pool)
        selected[source+'_train']=pool[:n]
    selected['E1']=selected['ChartQA_train']
    subset=list(selected['E1']);rng.shuffle(subset);selected['E2']=subset[:500]+selected['TabMWP_train']
    selected['E3']=selected['PGPS9K_train']
    ready=all(len(selected[k])==1000 for k in ['E1','E2','E3']) and all(len(selected[k+'_check'])==100 for k in ['ChartQA','TabMWP','PGPS9K']) and len(evaluation)==900
    target=Path(cfg['processed_dir']) if ready else run/'partial_processed'
    target.mkdir(parents=True,exist_ok=True)
    files={}
    for name,data in selected.items():
        path=target/(name+'.jsonl');content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data)
        if path.exists() and path.read_text()!=content:raise Blocked('Processed dataset would change; use a new processed_dir/version instead: '+str(path))
        path.write_text(content);files[name]=dict(path=str(path),count=len(data),sha256=sha(path))
    ready=all(len(selected[k])==1000 for k in ['E1','E2','E3']) and all(len(selected[k+'_check'])==100 for k in ['ChartQA','TabMWP','PGPS9K']) and len(evaluation)==900
    manifest=dict(target_dir=str(target),ready_for_training=ready,processing_version='P04_v1',seed=3407,files=files,sources={'ChartQA':CHART,'TabMWP':TAB,'PGPS9K':PGPS},rejected=rejected,perceptual_overlap_candidates=suspects,
        overlap_check=dict(mmmu_questions=len(questions),mmmu_image_check=bool(evaluation),method='exact normalized questions, decoded RGB pixels;dHash<=2 candidates conservatively excluded',semantic_duplicate_check=False),
        pgps_missing=pgps_error,pgps_geometry3k_image_groups=len(geometry),pgps_geometry3k_training_overlap=sum(r['pixel_sha256'] in geometry for r in selected['E3']),geometry3k_note='PGPS9K contains Geometry3K-derived data; not an independent geometry benchmark',task_labels=None,manual_review='see sample_review.json; automated QA/source linkage is not manual reasoning verification')
    save(target/'manifest.json',manifest);save(run/'data_manifest.json',manifest)
    counts={k:len(v) for k,v in selected.items()};save(run/'preparation_counts.json',counts)
    if not ready and not cfg['allow_partial']:raise Blocked('Incomplete datasets or MMMU overlap check; see manifest counts; no samples duplicated')
    return dict(counts=counts,ready_for_training=ready,partial=not ready,rejected_count=len(rejected))
