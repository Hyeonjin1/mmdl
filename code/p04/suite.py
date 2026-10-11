"""Host-only E1/E2 orchestrator: recorded sequential processes, no original re-inference."""
import argparse,fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from records import ROOT,save,sha,index,now,Blocked
from host_steps import create
from contracts import b0

STATE=ROOT/'runs/P04_E12_suite_state.json'
CURRENT=None

def signature(cfg):
    from contracts import digest
    files=['code/p04/engine.py','code/p04/evaluate_vllm.py','code/p04/contracts.py','code/p04/launch.py',
           'code/p04/scoring_latest_v2.py','code/p04/scoring_v2.py','code/p03/learning.py','code/p03/gpu_admission.py',
           'code/p04/suite.py','code/p04/test_pipeline.py','configs/P04/evaluation_requirements.txt']
    return digest({'config':cfg,'sources':{p:sha(ROOT/p) for p in files}})

def show():
    if not STATE.exists():print('E1/E2 suite has not started.');return
    state=json.loads(STATE.read_text());print(json.dumps(state,ensure_ascii=False,indent=2))
    active=state.get('active_run')
    if active:
        run=Path(active)
        for name in ['metrics.json','execution_phase.json']:
            if (run/name).is_file():print(name,(run/name).read_text())
        print('Log:',run/'logs/stage.log')

def stop():
    if not STATE.exists():raise Blocked('No suite PID recorded')
    state=json.loads(STATE.read_text());pid=state.get('pid')
    if not pid:raise Blocked('No active suite PID recorded')
    cmd=Path('/proc')/str(pid)/'cmdline'
    if not cmd.exists():print('Recorded suite is no longer running.');return
    if b'code/p04/suite.py' not in cmd.read_bytes():raise Blocked('Recorded PID is not this suite; no signal sent')
    os.kill(pid,signal.SIGTERM);print('SIGTERM sent to owned suite PID',pid)

def terminate(signum,frame):
    global CURRENT
    if CURRENT and CURRENT.poll() is None:
        #Only this suite's newly created stage process group, never unrelated GPU jobs.
        os.killpg(CURRENT.pid,signal.SIGTERM)
    raise KeyboardInterrupt('suite signal '+str(signum))

def main():
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--stop',action='store_true')
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--config',type=Path,default=ROOT/'configs/P04/E12_latest_vllm.json')
    args=p.parse_args()
    if args.status:show();return 0
    if args.stop:stop();return 0
    cfg=json.loads(args.config.read_text());run=create('E12_suite');cfg.update(stage='E12_suite');save(run/'config_resolved.json',cfg)
    (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/user/MMDL\nbash scripts/run_p04_e12.sh'+(' --retry-failed' if args.retry_failed else '')+'\n')
    (ROOT/'.locks').mkdir(exist_ok=True);lock=(ROOT/'.locks/p04-e12-suite.lock').open('a')
    result={};state={};start=time.perf_counter()
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,protocol,_=b0(cfg);stamp=signature(cfg)
        state=json.loads(STATE.read_text()) if STATE.exists() else {'completed':{},'history':[],'signature':stamp}
        if state['signature']!=stamp:raise Blocked('Suite source/settings signature changed. Preserve completed work; review contract before starting a new suite.')
        if state.get('failed_stage') and not args.retry_failed:raise Blocked('Prior stage failed; dependent stages remain stopped. Inspect failure, then explicitly use --retry-failed if conditions are unchanged.')
        state.update(pid=os.getpid(),suite_run=str(run),status='running');save(STATE,state)
        save(run/'data_manifest.json',{'B0_protocol_sha256':sha(Path(cfg['b0_run'])/'evaluation_protocol.json'),
            'processed_manifest_path':str(Path(cfg['processed_dir'])/'manifest.json'),'E3_required':False,'original_model_inference_allowed':False})
        signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
        def stage(key,name,experiment='B0',adapter=None,smoke=False,python=None,extra=None):
            global CURRENT
            if key in state['completed']:
                prior=Path(state['completed'][key]);m=json.loads((prior/'metrics.json').read_text())
                if m.get('status')!='completed':raise Blocked('Completed stage receipt no longer agrees:'+key)
                print('REUSE_COMPLETED',key,prior,flush=True);return prior
            rid=run.name+'_'+key;out=ROOT/'runs'/rid
            command=[python or str(ROOT/'.venv-p03/bin/python'),str(ROOT/'code/p04/launch.py'),'--config',str(args.config.resolve()),
               '--stage',name,'--experiment',experiment,'--run-id',rid]
            if adapter:command+=['--adapter',str(adapter)]
            if smoke:command+=['--smoke']
            if state.get('failed_stage')==key and args.retry_failed and name=='evaluate':
                failed=Path(state['failed_run'])
                if (failed/'evaluation_contract.json').is_file():command+=['--resume-from',str(failed)]
            command+=extra or []
            state.update(active_stage=key,active_run=str(out));save(STATE,state)
            print('START',key,out,flush=True)
            with (run/'logs'/('stage_'+key+'.log')).open('x') as log:
                CURRENT=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
                returncode=CURRENT.wait();CURRENT=None
            metrics=json.loads((out/'metrics.json').read_text()) if (out/'metrics.json').exists() else {'status':'missing'}
            state['history'].append({'key':key,'run':str(out),'status':metrics.get('status'),'returncode':returncode})
            if metrics.get('status')=='completed':state['completed'][key]=str(out)
            if returncode or metrics.get('status')!='completed':
                state.update(failed_stage=key,failed_run=str(out),status='failed');save(STATE,state)
                raise Blocked('Stage failed; subsequent experiments stopped:'+key+' '+str(out))
            if state.get('failed_stage')==key:state.pop('failed_stage',None);state.pop('failed_run',None)
            save(STATE,state);print('DONE',key,out,flush=True);return out
        #All dependencies before any training; never waits for/downloads PGPS9K or infers original B0.
        stage('ready','ready')
        eval_python=cfg['eval_python']
        if not Path(eval_python).is_file():raise Blocked('Separate evaluation Python not installed. Run bash scripts/setup_p04_eval_env.sh first.')
        stage('eval_envcheck','envcheck',python=eval_python)
        stage('synthetic_tests','selftest')
        smoke_train=stage('smoke_train','train','smoke',smoke=True)
        stage('smoke_reload','reload','smoke',adapter=smoke_train/'adapter',smoke=True)
        stage('smoke_eval','evaluate','smoke',adapter=smoke_train/'adapter',smoke=True,python=eval_python)
        evaluations={}
        for experiment in ['E1','E2']:
            trained=stage(experiment+'_train','train',experiment)
            stage(experiment+'_reload','reload',experiment,adapter=trained/'adapter')
            evaluations[experiment]=stage(experiment+'_eval','evaluate',experiment,adapter=trained/'adapter',python=eval_python)
        stage('comparison','analyze',extra=['--comparison-e1',str(evaluations['E1']),'--comparison-e2',str(evaluations['E2'])])
        state.update(status='completed',active_stage=None,active_run=None);save(STATE,state)
        result={'status':'completed','optimizer_steps':0,'completed_stages':state['completed'],'original_model_inference':False,'E3_required':False}
    except BaseException as e:
        result={'status':'interrupted' if isinstance(e,KeyboardInterrupt) else 'blocked' if isinstance(e,(Blocked,BlockingIOError)) else 'error',
                'error':repr(e),'optimizer_steps':0,'dependent_steps_stopped':True}
        (run/'logs/error.log').write_text(traceback.format_exc())
        #Do not replace another concurrently running suite's state after a lock rejection.
        if state.get('pid')==os.getpid():
            state['status']=result['status'];save(STATE,state)
        if CURRENT and CURRENT.poll() is None:
            try:CURRENT.wait(timeout=30)
            except subprocess.TimeoutExpired:
                (run/'logs/termination_pending.txt').write_text('Owned stage group remains alive after SIGTERM; inspect before retrying. No unrelated process was killed.\n')
    finally:lock.close()
    result.update(wall_seconds=time.perf_counter()-start,ended_utc=now());save(run/'metrics.json',result)
    with (run/'experiment.md').open('a') as f:f.write('\n## Outcome\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\nFailed dependencies stop follow-on stages. Completed stages are preserved and reused. All training starts independently from pinned original model; smoke adapter is never continued for main experiments. Original B0 is never inferred.\n')
    index();print(run,result['status'],flush=True);return result['status']!='completed'

if __name__=='__main__':raise SystemExit(main())
