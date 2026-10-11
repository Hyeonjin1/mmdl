"""Recorded stage launcher. No stage executes without its required records."""
import argparse,contextlib,fcntl,importlib,json,os,shlex,signal,sys,time,traceback
from pathlib import Path
from records import ROOT,save,sha,environment,provenance,index,now,Blocked

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['acquire','mmmu','prepare','ready','validate','parser_test','train','reload','evaluate','analyze','setup_eval','envcheck','selftest'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/P04/common.json')
    parser.add_argument('--experiment',choices=['B0','E1','E2','E3','smoke'],default='B0')
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--comparison-e1',type=Path);parser.add_argument('--comparison-e2',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run-id');parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    #environment() probes CUDA; vLLM workers must spawn rather than fork that state.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    cfg.update(stage=args.stage,experiment=args.experiment,adapter_path=str(args.adapter.resolve()) if args.adapter else None,smoke=args.smoke,allow_partial=args.allow_partial,
       resume_from=str(args.resume_from.resolve()) if args.resume_from else None)
    if args.comparison_e1 and args.comparison_e2:cfg['comparison_runs']={'E1':str(args.comparison_e1.resolve()),'E2':str(args.comparison_e2.resolve())}
    if args.resume_from and args.stage!='evaluate':parser.error('Only evaluation has verified completed-response resume')
    assert cfg['model_revision']=='ebb281ec70b05090aa6165b016eac8ec08e71b17' and cfg['world_size']==1
    if args.stage=='train' and args.adapter:parser.error('Every training run must start from unadapted original model')
    rid=args.run_id or 'P04_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+args.stage+'_'+args.experiment
    if Path(rid).name!=rid or not rid.startswith('P04_'):parser.error('Invalid run id')
    run=ROOT/'runs'/rid;run.mkdir(parents=True,exist_ok=False);(run/'logs').mkdir()
    save(run/'config_resolved.json',cfg);save(run/'code_provenance.json',provenance())
    save(run/'environment.json',{'status':'pending probe; mandatory record created before execution'})
    save(run/'data_manifest.json',{'status':'pending','evaluation_is_development_comparison':True})
    save(run/'metrics.json',{'status':'created','optimizer_steps':0,'started_utc':now()})
    command=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]
    if '--run-id' in command:
        i=command.index('--run-id');del command[i:i+2]
    (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join(command)+'\n')
    (run/'experiment.md').write_text(f'# {rid}\n\nPurpose: {cfg["purpose"]}\n\nConditions: config_resolved.json, environment.json, data_manifest.json and code_provenance.json.\n\nResult: pending.\n\nLimitations: single seed; MMMU validation is a development comparison set. No comparison to mixed563/900 baseline.\n\nNext: verify all data, masks, independent original initialization and fixed evaluation before conclusions.\n')
    status='completed';result={};start=time.perf_counter();lock=None
    def stop(sig,frame):raise KeyboardInterrupt('signal '+str(sig))
    signal.signal(signal.SIGTERM,stop)
    with (run/'logs/stage.log').open('x',buffering=1) as log:
        try:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                if args.stage in ['train','reload','evaluate']:
                    (ROOT/'.locks').mkdir(exist_ok=True);lock=(ROOT/'.locks/p04-gpu0.lock').open('a')
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    from gpu_admission import select_gpu
                    os.environ['CUDA_VISIBLE_DEVICES']='0';selection=select_gpu();save(run/'gpu_selection.json',selection)
                    save(run/'environment.json',environment())
                    if selection.get('error'):raise Blocked(selection['error'])
                else:save(run/'environment.json',environment())
                for name in ['experiment.md','config_resolved.json','environment.json','data_manifest.json','commands.sh','metrics.json','code_provenance.json']:
                    if not (run/name).is_file():raise Blocked('Required record missing:'+name)
                save(run/'metrics.json',{'status':'running','started_utc':now(),'optimizer_steps':0});index()
                if args.stage=='parser_test':
                    from scoring_latest_v2 import selftest
                    result=selftest();save(run/'parser_tests.json',result)
                elif args.stage=='acquire':
                    from acquire import acquire
                    result=acquire(run)
                elif args.stage in ['mmmu','prepare']:
                    from data import prepare,mmmu
                    result={'prepare':prepare,'mmmu':mmmu}[args.stage](cfg,run)
                elif args.stage=='ready':
                    from contracts import ready
                    result=ready(cfg,run)
                elif args.stage=='envcheck':
                    from evaluate_vllm import envcheck
                    result=envcheck(cfg,run)
                else:
                    from engine import validate,train,reload,evaluate,analyze,setup_eval,selftest
                    result={'validate':validate,'train':train,'reload':reload,'evaluate':evaluate,'analyze':analyze,'setup_eval':setup_eval,'selftest':selftest}[args.stage](cfg,run)
        except BaseException as exc:
            status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'blocked' if isinstance(exc,Blocked) else 'oom' if 'out of memory' in str(exc).lower() else 'error'
            result={'error':repr(exc)};save(run/'logs/error.json',dict(error=repr(exc),traceback=traceback.format_exc()));log.write(traceback.format_exc())
        finally:
            if lock:lock.close()
    for filename in ['progress.json','execution_phase.json']:
        if (run/filename).exists():result={**json.loads((run/filename).read_text()),**result}
    result.update(status=status,wall_seconds=time.perf_counter()-start,ended_utc=now())
    save(run/'metrics.json',result)
    with (run/'experiment.md').open('a') as f:f.write('\n## Outcome\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    if status!='completed':
        with (run/'experiment.md').open('a') as f:f.write('\nDependent steps are stopped. OOM location is the last recorded phase; no unrecorded condition changes or automatic retries.\n')
    index();print(run,status)
    import subprocess
    with (run/'logs/publication.log').open('a') as log:
        try:
            published=subprocess.run([sys.executable,str(ROOT/'code/reporting/reports.py'),'--run-id',rid,'--publish'],stdout=log,stderr=log,timeout=180)
            if published.returncode:save(run/'publication_hook_error.json',{'returncode':published.returncode,'separate_from_experiment_status':True})
        except Exception as exc:save(run/'publication_hook_error.json',{'error':repr(exc)})
    return 0 if status=='completed' else 1
if __name__=='__main__':sys.exit(main())
