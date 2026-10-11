"""Verified HTTP byte ranges into exclusive sparse files; no existing-file deletion."""
import concurrent.futures, hashlib, json, os, shutil, threading, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=Path((ROOT/'.restore/current_run.txt').read_text().strip())
REV='ebb281ec70b05090aa6165b016eac8ec08e71b17'
TARGET=Path('/dev/shm/mmdl-user')/REV
META=json.loads((Path.home()/'mmdl-preparation/experiments/exp002_current_host_baseline_audit/results/student_metadata.json').read_text())
CHUNK=32*1024*1024
LOCK=threading.Lock()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while c:=f.read(4*1024*1024):h.update(c)
    return h.hexdigest()

def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    weights=[f for f in META['files'] if f['rfilename'].endswith('.safetensors')]
    need=sum(f['size'] for f in weights if not (TARGET/f['rfilename']).exists())
    if shutil.disk_usage(TARGET).free<need+2*2**30:raise RuntimeError('RAM filesystem reserve insufficient')
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if v.strip().endswith('kB')}
    if mem['MemAvailable']<need+10*2**30:raise RuntimeError('Insufficient memory reserve for weights plus model loading')
    fds={};partials={};jobs=[];completed=0;total=need;started=time.monotonic()
    progress=ROOT/'.restore/model_progress.json'
    for meta in weights:
        final=TARGET/meta['rfilename']
        if final.exists():
            if final.stat().st_size!=meta['size'] or sha(final)!=meta['lfs']['sha256']:raise RuntimeError('Existing complete model shard invalid; preserved')
            continue
        tmp=final.with_name(final.name+'.ranges_'+str(time.time_ns()))
        fd=os.open(tmp,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.ftruncate(fd,meta['size']);fds[meta['rfilename']]=fd;partials[meta['rfilename']]=tmp
        for start in range(0,meta['size'],CHUNK):jobs.append((meta,start,min(meta['size']-1,start+CHUNK-1)))
    def download(job):
        nonlocal completed
        meta,start,end=job;url=f"https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/{REV}/{meta['rfilename']}"
        last=None
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f"bytes {start}-{end}/{meta['size']}":raise RuntimeError('Server did not honor exact range')
                    offset=start
                    while data:=response.read(1024*1024):
                        if offset+len(data)>end+1:raise RuntimeError('Range overflow')
                        n=os.pwrite(fds[meta['rfilename']],data,offset)
                        if n!=len(data):raise RuntimeError('Short range write')
                        offset+=n
                    if offset!=end+1:raise RuntimeError('Incomplete HTTP range')
                with LOCK:
                    completed+=end-start+1
                    record={'completed_bytes':completed,'total_bytes':total,'elapsed_seconds':time.monotonic()-started,'workers':16,'source_revision':REV,'RAM_cache_volatile':True}
                    temporary=progress.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(progress)
                    with (RUN/'logs/model_ranges.jsonl').open('a') as log:log.write(json.dumps({'file':meta['rfilename'],'start':start,'end':end,'completed':True})+'\n')
                return
            except Exception as e:last=e
        raise last
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(download,jobs))
    finally:
        for fd in fds.values():os.close(fd)
    for meta in weights:
        if meta['rfilename'] not in partials:continue
        temporary=partials[meta['rfilename']]
        if sha(temporary)!=meta['lfs']['sha256']:raise RuntimeError('Full shard checksum mismatch; partial preserved')
        temporary.rename(TARGET/meta['rfilename'])
    processor=ROOT/'datasets/models'/REV
    for meta in META['files']:
        if not meta['rfilename'].endswith(('.json','.txt')):continue
        source=processor/meta['rfilename'];dest=TARGET/meta['rfilename']
        if dest.exists() and sha(dest)!=sha(source):raise RuntimeError('Existing processor file differs')
        if not dest.exists():
            with source.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
    files=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(TARGET.iterdir()) if p.name in {m['rfilename'] for m in META['files']}]
    record={'model':'Qwen/Qwen3-VL-4B-Instruct','revision':REV,'complete':True,'volatile_RAM_storage':True,'files':files}
    for name in ['model_manifest.json','processor_manifest.json']:
        with (TARGET/name).open('x') as f:json.dump(record,f,indent=2)
    (RUN/'model_manifest.json').write_text(json.dumps(record,indent=2))
    cfgpath=ROOT/'configs/restore/E123.json';cfg=json.loads(cfgpath.read_text());cfg['model_path']=str(TARGET);cfg['model_storage_volatile']=True;cfgpath.write_text(json.dumps(cfg,indent=2)+'\n')
    print('MODEL_COMPLETE',TARGET,'all weight/processor files verified',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        with (RUN/('model_parallel_failure_'+str(time.time_ns())+'.json')).open('x') as f:json.dump({'error':repr(error),'partial_files_preserved':True},f)
        raise
