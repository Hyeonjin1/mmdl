"""Pinned official assets; preserve existing files and record every download."""
import argparse, ast, concurrent.futures, hashlib, io, json, os, shutil, sys, time, urllib.parse, urllib.request, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/p04'),str(ROOT/'code/p03')]
from acquire import CHART,TAB,PGPS
from data import image_info
RAW=ROOT/'datasets/raw'
STATE=ROOT/'.restore'
RUN=Path((STATE/'current_run.txt').read_text().strip())
RESERVE=2*2**30
LOG=RUN/'logs/downloads.jsonl'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text()!=content:raise RuntimeError('Existing record differs: '+str(path))
    if not path.exists():path.write_text(content)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()
def fetch(url,path,meta=None,reuse=None):
    path=Path(path);meta=meta or {}
    def valid(p):
        return p.is_file() and (not meta.get('size') or p.stat().st_size==meta['size']) and (not meta.get('lfs') or sha(p)==meta['lfs']['sha256'])
    if path.exists():
        if not valid(path):raise RuntimeError('Existing asset invalid; preserved: '+str(path))
    elif reuse and valid(Path(reuse)):
        path.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(reuse,path)
        except OSError as error:
            import errno
            if error.errno!=errno.EXDEV:raise
            with Path(reuse).open('rb') as src,path.open('xb') as dst:shutil.copyfileobj(src,dst)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.attempt_'+str(time.time_ns()))
        with urllib.request.urlopen(url,timeout=60) as response:
            expected=int(response.headers.get('Content-Length',meta.get('size',0)))
            if shutil.disk_usage(ROOT).free<expected+RESERVE:raise RuntimeError('Insufficient disk reserve before '+str(path))
            with temporary.open('xb') as f:
                while chunk:=response.read(1024*1024):
                    if shutil.disk_usage(ROOT).free<len(chunk)+RESERVE:raise RuntimeError('Disk reserve reached; partial download preserved')
                    f.write(chunk)
        if not valid(temporary):raise RuntimeError('Size/hash mismatch; partial preserved: '+str(temporary))
        if path.exists():raise RuntimeError('Concurrent destination appeared: '+str(path))
        temporary.rename(path)
    record={'url':url,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'revision_from_url':True}
    with LOG.open('a') as f:f.write(json.dumps(record)+'\n')
    return record
def rawurl(repo,rev,name):return 'https://raw.githubusercontent.com/'+repo+'/'+rev+'/'+urllib.parse.quote(name)
def parallel(jobs):
    errors=[];done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        tasks={pool.submit(fetch,*job):job for job in jobs}
        for future in concurrent.futures.as_completed(tasks):
            try:done.append(future.result())
            except Exception as e:errors.append({'url':tasks[future][0],'error':repr(e)})
    if errors:
        write(RUN/('download_errors_'+str(time.time_ns())+'.json'),errors)
        raise RuntimeError(f'{len(errors)} downloads failed; successful files preserved')
    return done

def mmmu():
    revision='98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
    old=Path.home()/'mmdl-preparation/experiments/exp002_current_host_baseline_audit'
    metadata=json.loads((old/'results/mmmu_pinned_metadata.json').read_text())
    files=[f for f in metadata['files'] if '/validation-' in f['rfilename']]
    required=sum(f['size'] for f in files)
    if shutil.disk_usage(ROOT).free<required*3+RESERVE:raise RuntimeError('MMMU acquisition budget exceeds free space')
    paths=parallel([(f'https://huggingface.co/datasets/MMMU/MMMU/resolve/{revision}/'+f['rfilename'],
        RAW/'MMMU_restore/parquet'/f['rfilename'],f,old/'data/parquet'/f['rfilename']) for f in files])
    import pyarrow.parquet as pq
    from PIL import Image
    source=ROOT/'review_snapshots/0c0b415f6f93923a4249d3da32c5cbb447a518ea/results/validation900_cot8192_guard_full'
    predictions=[json.loads(line) for line in (source/'predictions.jsonl').read_text().splitlines()]
    rc=json.loads((source/'run_config.json').read_text())
    samples={}
    for file in paths:
        for row in pq.read_table(file['path']).to_pylist():samples[row['id']]=row
    out=[];imagechecks=[];target=RAW/'MMMU_latest_vllm';(target/'images').mkdir(parents=True,exist_ok=True)
    for pred in predictions:
        item=samples[pred['question_id']];ann=pred['annotation']
        if any(item[k]!=ann[k] for k in ['question','answer','question_type']):raise ValueError('Historical QA differs: '+item['id'])
        opts=ast.literal_eval(item['options']) if isinstance(item['options'],str) else item['options']
        if ann['question_type']=='multiple-choice' and opts!=[ann[k] for k in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if ann.get(k) is not None]:raise ValueError('Options differ')
        row={k:ann[k] for k in ['index','question','answer','split','category','question_type']}
        if ann['question_type']=='multiple-choice':row.update({chr(65+i):value for i,value in enumerate(opts)})
        names=[];hashes=[]
        for i in range(1,8):
            value=item.get('image_'+str(i))
            if value is None:continue
            image=Image.open(io.BytesIO(value['bytes'])) if value.get('bytes') else Image.open(value['path'])
            dest=target/'images'/(item['id']+'_'+str(i)+'.png');buffer=io.BytesIO();image.convert('RGB').save(buffer,format='PNG')
            content=buffer.getvalue()
            if dest.exists() and dest.read_bytes()!=content:raise ValueError('Existing RGB export differs')
            if not dest.exists():dest.write_bytes(content)
            names.append('images/'+dest.name);hashes.append(sha(dest));imagechecks.append({'id':item['id'],**image_info(dest)})
        if len(names)!=len(ann['image_path']):raise ValueError('Historical image count differs')
        row.update(image_path=names,image_sha256=hashes);out.append(row)
    out.sort(key=lambda r:(r['category'],int(r['index'].rsplit('_',1)[1])))
    manifest=target/'validation.jsonl';content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out)
    if manifest.exists() and manifest.read_text()!=content:raise ValueError('Existing manifest differs')
    if not manifest.exists():manifest.write_text(content)
    matched=sha(manifest)==rc['dataset_sha256']
    write(target/'provenance.json',{'source':'MMMU/MMMU','revision':revision,'split':'validation','count':len(out),'image_count':len(imagechecks),
          'manifest_sha256':sha(manifest),'historical_manifest_sha256':rc['dataset_sha256'],'historical_manifest_matches':matched,
          'image_policy':'RGB PNG original image_N ascending','training_use':False,'files':paths})
    write(RUN/'mmmu_image_checks.json',imagechecks)
    if not matched:raise ValueError('Exact historical manifest hash mismatch; no matched evaluation claim')
    print('MMMU_READY',len(out),len(imagechecks),sha(manifest),flush=True)

def chart_tab():
    if shutil.disk_usage(ROOT).free<2*2**30+RESERVE:raise RuntimeError('Chart/Tab conservative 2GiB acquisition budget unavailable')
    jobs=[]
    for split in ['train','val']:
        name=split+'_human.json';fetch(rawurl('vis-nlp/ChartQA',CHART,f'ChartQA Dataset/{split}/{name}'),RAW/'ChartQA_P04'/name)
    for name in ['problems_train.json','problems_dev1k.json','splits.json']:
        fetch(rawurl('lupantech/PromptPG',TAB,'data/tabmwp/'+name),RAW/'TabMWP_P04'/name)
    # Published selection algorithm; new manifests are explicitly a new version.
    from acquire import candidates
    _,requests=candidates()
    for request in requests:jobs.append((rawurl(request['repo'],request['revision'],request['remote']),request['local']))
    parallel(jobs)
    for repo,rev,folder,license in [('vis-nlp/ChartQA',CHART,'ChartQA_P04','LICENSE'),('lupantech/PromptPG',TAB,'TabMWP_P04','LICENSE.md')]:
        for name in ['README.md',license]:fetch(rawurl(repo,rev,name),RAW/folder/'source'/name)
    print('CHART_TAB_READY',len(requests),'image/table files',flush=True)

def pgps():
    metadata=json.loads((STATE/'pgps.json').read_text());meta=next(f for f in metadata['siblings'] if f['rfilename']=='PGPS9K.zip')
    target=RAW/'PGPS9K_P04';archive=target/'PGPS9K.zip'
    fetch(f'https://huggingface.co/datasets/PeijieWang/PGPS9K/resolve/{PGPS}/PGPS9K.zip',archive,meta)
    with zipfile.ZipFile(archive) as z:
        unpacked=sum(i.file_size for i in z.infolist())
        if shutil.disk_usage(ROOT).free<unpacked+RESERVE:raise RuntimeError('PGPS extraction budget unavailable')
        for info in z.infolist():
            dest=(target/'extracted'/info.filename).resolve()
            if not dest.is_relative_to((target/'extracted').resolve()):raise ValueError('Unsafe ZIP path')
            if info.is_dir():continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            content=z.read(info,pwd=b'PAL_PGPS_2023')
            if dest.exists() and dest.read_bytes()!=content:raise ValueError('Existing extraction differs')
            if not dest.exists():dest.write_bytes(content)
    fetch(f'https://huggingface.co/datasets/PeijieWang/PGPS9K/resolve/{PGPS}/README.md',target/'source/README.md')
    write(RUN/'pgps_extraction.json',{'revision':PGPS,'archive_sha256':sha(archive),'compressed_bytes':archive.stat().st_size,'uncompressed_bytes':unpacked,'file_count':len(z.infolist()),'password_source':'official PGPS README','complete':True})
    print('PGPS_READY',unpacked,flush=True)

def model(processor_only=False,target_override=None):
    old=Path.home()/'mmdl-preparation/experiments/exp002_current_host_baseline_audit'
    meta=json.loads((old/'results/student_metadata.json').read_text());files=[f for f in meta['files'] if f['rfilename'].endswith(('.json','.txt')) or (not processor_only and f['rfilename'].endswith('.safetensors'))]
    target=Path(target_override) if target_override else ROOT/'datasets/models'/meta['sha']
    target.parent.mkdir(parents=True,exist_ok=True)
    required=sum(f['size'] for f in files if not (target/f['rfilename']).exists())
    free=shutil.disk_usage(target.parent).free
    write(RUN/('model_budget_processor.json' if processor_only else 'model_budget_weights.json'),{'remaining_download_bytes':required,'storage_free_bytes':free,'target':str(target),'reserve_bytes':RESERVE,'processor_only':processor_only,'volatile_storage':str(target).startswith('/dev/shm/')})
    if free<required+RESERVE:raise RuntimeError('Model download exceeds current storage reserve; provide additional storage')
    result=sorted(parallel([(f"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/{meta['sha']}/"+f['rfilename'],target/f['rfilename'],f,old/'data/student_processor'/f['rfilename']) for f in files]),key=lambda r:r['path'])
    write(target/('processor_manifest.json' if processor_only else 'model_manifest.json'),{'model':'Qwen/Qwen3-VL-4B-Instruct','revision':meta['sha'],'processor_only':processor_only,'files':result})
    print('MODEL_PROCESSOR_READY' if processor_only else 'MODEL_READY',target,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mmmu','chart_tab','pgps','processor','model']);p.add_argument('--target');a=p.parse_args()
    try:
        {'mmmu':mmmu,'chart_tab':chart_tab,'pgps':pgps,'processor':lambda:model(True),'model':lambda:model(False,a.target)}[a.stage]()
    except BaseException as e:
        write(RUN/('asset_failure_'+a.stage+'_'+str(time.time_ns())+'.json'),{'stage':a.stage,'error':repr(e),'completed_assets_preserved':True});raise
