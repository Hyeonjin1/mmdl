"""Pinned official data downloads; TLS verification stays enabled."""
import hashlib,json,random,re,shutil,zipfile
from pathlib import Path
from urllib.request import urlopen,Request
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'datasets/raw'
CHART='044eabfc306abfe9340c5741f0093aefc5973d06'
TAB='5a1a52214521590b075f545b76a4f5ce666345e3'
PGPS='7dfd9dad4e455344b751b87340cd0ecf8798f4db'
CALC=re.compile(r'difference|sum|total|ratio|percent|average|mean|median|more|less|fewer|many|greater|least|most|higher|lower|combined|times|range',re.I)

def download(url,path):
    path=Path(path)
    if path.is_file(): return path
    if shutil.disk_usage(ROOT).free<4*2**30: raise RuntimeError('Less than4GiB disk reserve; download blocked')
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with urlopen(Request(url,headers={'User-Agent':'MMDL-experiment-data'}),timeout=45) as response, partial.open('wb') as out:
        shutil.copyfileobj(response,out)
    partial.rename(path);return path

def url(repo,revision,path):return f'https://raw.githubusercontent.com/{repo}/{revision}/'+quote(path)

def candidates():
    rng=random.Random(3407); result={}; requests=[]
    chart=RAW/'ChartQA_P04';chart.mkdir(exist_ok=True)
    old=RAW/'ChartQA_p03/train_human.json'
    if old.is_file() and not (chart/'train_human.json').exists():shutil.copy2(old,chart/'train_human.json')
    for split,n in [('train',1100),('val',140)]:
        p=chart/(split+'_human.json')
        if not p.exists():download(url('vis-nlp/ChartQA',CHART,f'ChartQA Dataset/{split}/{split}_human.json'),p)
        rows=[]
        for i,row in enumerate(json.loads(p.read_text())):
            if CALC.search(row['query']) and re.fullmatch(r'[+-]?[\d,.]+%?',str(row['label']).strip()):
                rows.append(dict(id=f'ChartQA:{split}:{i}',source='ChartQA',split=split,question=row['query'],answer=str(row['label']).strip(),options=[],original_index=i,chart_name=row['imgname']))
        rng.shuffle(rows);rows=rows[:n];result['ChartQA_'+split]=rows
        for row in rows:
            row['image']=str(chart/split/'png'/row['chart_name'])
            row['review_table']=str(chart/split/'tables'/Path(row['chart_name']).with_suffix('.csv'))
            for kind,path in [('png',Path(row['image'])),('tables',Path(row['review_table']))]:
                oldimage=RAW/'ChartQA_p03/png'/row['chart_name']
                if kind=='png' and split=='train' and oldimage.exists() and not path.exists():path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(oldimage,path)
                requests.append(dict(repo='vis-nlp/ChartQA',revision=CHART,remote=f'ChartQA Dataset/{split}/{kind}/{path.name}',local=str(path),binary=kind=='png'))
    tab=RAW/'TabMWP_P04'
    for name in ['problems_train.json','problems_dev1k.json','splits.json']:
        download(url('lupantech/PromptPG',TAB,'data/tabmwp/'+name),tab/name)
    splits=json.loads((tab/'splits.json').read_text())
    for split,n,filename in [('train',600,'problems_train.json'),('dev',140,'problems_dev1k.json')]:
        rows=[]
        allowed=set(map(str,splits[split]))
        for key,row in json.loads((tab/filename).read_text()).items():
            if key not in allowed:raise ValueError('TabMWP official split mismatch')
            if row['ques_type']=='free_text' and row['ans_type'] in ['integer_number','decimal_number'] and CALC.search(row['question']):
                rows.append(dict(id='TabMWP:'+key,source='TabMWP',split=split,question=row['question'],answer=str(row['answer']),unit=row.get('unit'),options=[],image=str(tab/'tables'/(key+'.png')),original_key=key,review_annotation=filename,table_text=row['table'],solution=row['solution']))
        rng.shuffle(rows);rows=rows[:n];result['TabMWP_'+split]=rows
        for row in rows:requests.append(dict(repo='lupantech/PromptPG',revision=TAB,remote='data/tabmwp/tables/'+Path(row['image']).name,local=row['image'],binary=True))
    (ROOT/'configs/P04/candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    requests=list({r['local']:r for r in requests}.values())
    (ROOT/'configs/P04/download_requests.json').write_text(json.dumps(requests,indent=2))
    return result,requests

def acquire(run):
    errors=[]
    for repo,rev,directory,license_name in [('vis-nlp/ChartQA',CHART,'ChartQA_P04','LICENSE'),('lupantech/PromptPG',TAB,'TabMWP_P04','LICENSE.md')]:
        for name in ['README.md',license_name]:
            try:download(url(repo,rev,name),RAW/directory/'source'/name)
            except Exception as e:errors.append(dict(url=url(repo,rev,name),error=repr(e)))
    try:
        rows,requests=candidates()
        for request in requests:
            try:download(url(request['repo'],request['revision'],request['remote']),request['local'])
            except Exception as e:errors.append(dict(file=request['local'],error=repr(e)))
    except Exception as e:errors.append(dict(stage='candidate_acquisition',error=repr(e)))
    pgps=RAW/'PGPS9K_P04';archive=pgps/'PGPS9K.zip'
    try:
        download(f'https://huggingface.co/datasets/PeijieWang/PGPS9K/resolve/{PGPS}/PGPS9K.zip',archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!='a74b2b819ed4dff91e870b17a33b7860de5f118b254c1f48e8be0fe10ab9bc21':raise ValueError('Official PGPS9K archive SHA256 mismatch')
        with zipfile.ZipFile(archive) as z:
            total=sum(i.file_size for i in z.infolist())
            if total+4*2**30>shutil.disk_usage(ROOT).free:raise RuntimeError('PGPS9K extraction exceeds disk reserve')
            for item in z.infolist():
                target=(pgps/'extracted'/item.filename).resolve()
                if not target.is_relative_to((pgps/'extracted').resolve()):raise ValueError('Unsafe archive path')
                if item.is_dir() or target.exists():continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(item,pwd=b'PAL_PGPS_2023') as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    except Exception as e:errors.append(dict(source='PGPS9K official HF archive',error=repr(e)))
    (run/'download_errors.json').write_text(json.dumps(errors,indent=2))
    files={str(p.relative_to(RAW)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for d in ['ChartQA_P04','TabMWP_P04','PGPS9K_P04'] for p in (RAW/d).rglob('*') if p.is_file()}
    manifest=dict(status='downloaded' if not errors else 'incomplete',files=files,sources={'ChartQA':CHART,'TabMWP':TAB,'PGPS9K':PGPS},conditions={'ChartQA':'Repository GPL-3.0; original chart sources retain their rights','TabMWP':'Repository MIT; preserve dataset attribution','PGPS9K':'HF dataset card MIT; includes Geometry3K-derived examples'})
    (run/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    if errors:raise RuntimeError('Dataset acquisition incomplete; see download_errors.json')
    return dict(downloaded_files=len(files))
