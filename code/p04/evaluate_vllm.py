"""Adapter-only evaluation with frozen recent-vLLM input/generation/guard contract."""
import ast,copy,importlib.metadata,inspect,json,os,sys,time,threading,subprocess
from pathlib import Path
from records import ROOT,save,sha,rows,Blocked
from contracts import b0,image_manifest,checkpoint,digest
from rescore_latest import SNAP,aggregate
import scoring_latest_v2 as scorer
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'

SOURCE=SNAP/'code/MMDL-official-vllm'
ESSENTIAL={'vllm':'0.11.2','transformers':'4.57.1','qwen-vl-utils':'0.0.14','torch':'2.9.0','pillow':'12.3.0','numpy':'2.2.6','pandas':'2.3.3'}

def versions():
    result={}
    for package in ESSENTIAL:
        try:result[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:result[package]=None
    return result

def envcheck(cfg,run):
    _,protocol,_=b0(cfg);v=versions();errors=[]
    for name,expected in ESSENTIAL.items():
        #torch distribution may expose the CUDA suffix while original metadata did not.
        actual=v[name].split('+')[0] if v[name] else None
        if actual!=expected:errors.append({'package':name,'expected':expected,'actual':v[name]})
    save(run/'evaluation_environment_check.json',{'versions':v,'errors':errors,'original_recorded_packages':protocol['environment']})
    if errors:raise Blocked('Evaluation package versions differ from recent B0: '+json.dumps(errors))
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vllm.lora.utils import parse_fine_tuned_lora_name
    if 'lora_request' not in inspect.signature(AsyncLLM.generate).parameters:raise Blocked('Pinned AsyncLLM lacks adapter request support')
    mapper=Qwen3VLForConditionalGeneration.hf_to_vllm_mapper
    # Test real vLLM utility on every actual language LoRA name from successful P03, not guessed module suffixes.
    from safetensors import safe_open
    adapter=ROOT/'runs/P03_20261008T075816Z_train_bf16/adapter/adapter_model.safetensors'
    mapped=[]
    if not adapter.exists():raise Blocked('P03 adapter required for real saved-name compatibility check')
    with safe_open(str(adapter),framework='pt',device='cpu') as f:
        for key in f.keys():
            name,_=parse_fine_tuned_lora_name(key,weights_mapper=mapper)
            if not name.startswith('language_model.model.layers.') or any(s in name for s in ['visual','merger']):raise Blocked('Real LoRA mapping unexpected:'+name)
            mapped.append({'saved_key':key,'mapped_module':name})
    save(run/'vllm_lora_mapping_check.json',{'mapped_actual_tensor_keys':mapped,'count':len(mapped),
      'source':'vLLM0.11.2 Qwen3VL hf_to_vllm_mapper and real P03 saved tensor names'})
    return {'evaluation_environment_verified':True,'actual_lora_names_mapped':len(mapped),'versions':v,'GPU_inference_verified':False}

def pinned_functions(globals_):
    path=SOURCE/'upstream/evaluation/mmmu/run_mmmu.py'
    if sha(path)!='f9e4a6c1b8d2ad1709fb08dd46102e5f6e3423ada14df8df987743cf71b8eba0':
        raise Blocked('Pinned upstream prompt/preprocessor source changed')
    tree=ast.parse(path.read_text());names={'build_mmmu_prompt','prepare_inputs_for_vllm'}
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(body)!=2:raise Blocked('Pinned upstream functions unavailable')
    # Execute only the exact function ASTs; avoid importing GPT/Judge or another model backend.
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),globals_)
    return globals_['build_mmmu_prompt'],globals_['prepare_inputs_for_vllm']

class AdapterOnlyBackend:
    """Inject adapter request while preserving the original streaming/abort implementation."""
    def __init__(self,engine,request):self.engine=engine;self.request=request;self.submissions=0
    def generate(self,prompt,params,request_id):
        self.submissions+=1
        return self.engine.generate(prompt,params,request_id,lora_request=self.request)
    async def abort(self,request_id):return await self.engine.abort(request_id)
    def shutdown(self):return self.engine.shutdown()

class DeviceMemoryMonitor:
    """Device used memory, including parent/worker/display; not allocator peak stats."""
    def __init__(self,run):
        self.path=run/'logs/nvidia_memory.jsonl';self.stop_event=threading.Event();self.peak=None;self.thread=threading.Thread(target=self.poll,daemon=True)
    def poll(self):
        with self.path.open('a',buffering=1) as log:
            while not self.stop_event.is_set():
                try:
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
                    p=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    if p.returncode==0:
                        uuid,used,total=[v.strip() for v in p.stdout.strip().split(',')]
                        value=float(used);self.peak=value if self.peak is None else max(self.peak,value)
                        log.write(json.dumps({'elapsed_wall':time.time(),'uuid':uuid,'used_mib':value,'total_mib':float(total)})+'\n')
                except (OSError,ValueError,subprocess.TimeoutExpired):pass
                self.stop_event.wait(1)
    def start(self):self.thread.start()
    def close(self):self.stop_event.set();self.thread.join(timeout=6)

def evaluation_samples(cfg,protocol,source,manifest_path,manifest_rows):
    byid={r['index']:r for r in manifest_rows};samples=[]
    for old in source:
        ann=byid[old['question_id']];messages=copy.deepcopy(old['messages']);local_images=[str(manifest_path.parent/name) for name in ann['image_path']]
        items=[c for m in messages for c in m['content'] if c['type']=='image']
        for item,path in zip(items,local_images):item['image']=path
        samples.append({'id':old['question_id'],'dataset':'MMMU','subject':ann['category'],'question_type':ann['question_type'],
          'gold':ann['answer'],'options':{k:ann[k] for k in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if k in ann},'messages':messages,
          'expected_input_tokens':old['input_tokens'],'image_hashes':ann['image_sha256']})
    for dataset in ['ChartQA','TabMWP']:
        for row in rows(Path(cfg['processed_dir'])/(dataset+'_check.jsonl')):
            if sha(Path(row['image']))!=row['image_sha256']:raise Blocked('Heldout image changed:'+row['id'])
            messages=[{'role':'user','content':[{'type':'image','image':row['image'],'min_pixels':1003520,'max_pixels':4014080},
                        {'type':'text','text':'Question: '+row['question']+protocol['prompt_protocol']['cot_suffix']}]}]
            samples.append({'id':row['id'],'dataset':dataset,'subject':dataset,'question_type':'open','gold':row['answer'],'options':{},
              'messages':messages,'expected_input_tokens':None,'image_hashes':[row['image_sha256']]})
    if cfg.get('smoke'):
        mmmu=[r for r in samples if r['dataset']=='MMMU'];selection=[];seen=set()
        # Multiple subjects, MC/open and actual multi-image rows, chosen without GT or correctness.
        for pool in [[r for r in mmmu if r['question_type']=='open'],[r for r in mmmu if len(r['image_hashes'])>1],mmmu]:
            count=0
            for row in pool:
                if row['id'] not in seen:selection.append(row);seen.add(row['id']);count+=1
                if count==(2 if pool is not mmmu else 8):break
        for dataset in ['ChartQA','TabMWP']:selection.extend([r for r in samples if r['dataset']==dataset][:2])
        samples=selection
    return samples

def evaluate(cfg,run):
    if not cfg.get('adapter_path'):raise Blocked('Original-model re-inference is forbidden; provide completed adapter')
    envcheck(cfg,run);path,checkpoint_manifest,_=checkpoint(cfg);b0run,protocol,source=b0(cfg);mpath,mrows=image_manifest(cfg)
    if checkpoint_manifest['B0_protocol_sha256']!=sha(b0run/'evaluation_protocol.json'):raise Blocked('Adapter training protocol identity differs')
    if checkpoint_manifest.get('smoke') and not cfg.get('smoke'):raise Blocked('Smoke adapter cannot serve as main E1/E2 result')
    import torch,pandas as pd
    from transformers import AutoProcessor
    from huggingface_hub import snapshot_download
    from qwen_vl_utils import process_vision_info
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from learning import record_phase
    sys.path.insert(0,str(SOURCE));from streaming_guard import StreamingGuardEngine
    for name,expected in protocol['checkpoint_contract']['guard_code_sha256'].items():
        if sha(SOURCE/name)!=expected:raise Blocked('Historical repetition guard source differs')
    _,prepare=pinned_functions({'pd':pd,'process_vision_info':process_vision_info})
    model_path=snapshot_download(cfg['model_id'],revision=cfg['model_revision'],local_files_only=True)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    samples=evaluation_samples(cfg,protocol,source,mpath,mrows)
    engine_args=dict(protocol['engine_args']);engine_args['model']=model_path
    engine_args.update(enable_lora=True,max_loras=1,max_lora_rank=16,lora_dtype='bfloat16')
    settings=protocol['settings'];params=SamplingParams(**{k:settings[k] for k in ['temperature','top_p','top_k','repetition_penalty','presence_penalty']},
          max_tokens=8192,seed=3407,stop_token_ids=[])
    contract={'B0_protocol_sha256':sha(b0run/'evaluation_protocol.json'),'checkpoint_manifest_sha256':sha(path.parent/'checkpoint_manifest.json'),
       'MMMU_manifest_sha256':sha(mpath),'engine_args':engine_args,'adapter_request_path':str(path),'ids':[r['id'] for r in samples],
       'versions':versions(),'evaluation_code_sha256':sha(Path(__file__)),'sample_code_sha256':sha(ROOT/'code/p04/contracts.py'),
       'heldout_manifest_sha256':sha(Path(cfg['processed_dir'])/'manifest.json')}
    contract_hash=digest(contract);save(run/'evaluation_contract.json',dict(contract,hash=contract_hash))
    save(run/'data_manifest.json',{'MMMU_manifest':str(mpath),'MMMU_manifest_sha256':sha(mpath),'evaluation_ids':[r['id'] for r in samples],
      'checkpoint_manifest':checkpoint_manifest,'B0_run':str(b0run),'dataset_counts':dict(__import__('collections').Counter(r['dataset'] for r in samples))})
    results=[];prior=cfg.get('resume_from')
    if prior:
        prior=Path(prior);previous=json.loads((prior/'evaluation_contract.json').read_text())
        if previous['hash']!=contract_hash:raise Blocked('Resume contract differs; completed responses will not be reused')
        results=rows(prior/'responses.jsonl') if (prior/'responses.jsonl').exists() else []
        if len({r['id'] for r in results})!=len(results) or not {r['id'] for r in results}.issubset(contract['ids']):raise Blocked('Resume ID mismatch')
    response_path=run/'responses.jsonl';response_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    completed={r['id'] for r in results};engine=None;started=time.perf_counter();monitor=DeviceMemoryMonitor(run);monitor.start()
    try:
        if len(completed)<len(samples):
            record_phase(run,'vllm_engine_loading',0)
            engine=StreamingGuardEngine(engine_args);request=LoRARequest(cfg['experiment']+'_adapter',1,str(path))
            engine.engine=AdapterOnlyBackend(engine.engine,request)
            save(run/'adapter_runtime.json',{'enable_lora':True,'LoRARequest':{'name':request.lora_name,'id':1,'path':str(path)},
              'base_inference_requests':0,'lora_dtype':'bfloat16','language_only':True,
              'comparison_deviation':'Dynamic LoRA kernels/adapter enablement required for adapted weights; original sampling/prompt/image/guard policy retained. Bitwise backend equivalence is not guaranteed.'})
        for sample in samples:
            if sample['id'] in completed:continue
            start=time.perf_counter();record_phase(run,'vllm_preprocess',0,question_id=sample['id'])
            prepared=prepare(sample['messages'],processor)
            #Exact tokenizer/image token expansion count checked before generating any adapter answer.
            cpu=processor(text=[prepared['prompt']],images=prepared['multi_modal_data'].get('image'),return_tensors='pt',padding=False,truncation=False)
            count=int(cpu['input_ids'].shape[1]);grid=cpu.get('image_grid_thw');grids=grid.tolist() if grid is not None else []
            if sample['expected_input_tokens'] is not None and count!=sample['expected_input_tokens']:
                raise Blocked(f"Historical image/template input token mismatch:{sample['id']} expected{sample['expected_input_tokens']} actual{count}")
            if count+8192>49152:raise Blocked('Context budget does not fit; no silent truncation:'+sample['id'])
            del cpu
            record_phase(run,'vllm_generate_adapter',0,question_id=sample['id'])
            output=engine.generate([prepared],sampling_params=params,use_tqdm=False)[0];answer=output.outputs[0]
            if answer.finish_reason not in ['stop','length','repetition_abort']:raise RuntimeError('Unfinished adapter request')
            actual_input=len(output.prompt_token_ids or [])
            if actual_input!=count:raise Blocked('vLLM and CPU processor token expansion differ; no comparable score')
            text=answer.text.split('</think>')[-1].strip();raw_e=scorer.extract(text,sample['options']);aborted=answer.finish_reason=='repetition_abort'
            e=raw_e if not aborted else {'answer':None,'method':'repetition_abort_policy','failure':None}
            correct=not aborted and scorer.judge(e,sample['gold'],sample['options'])
            row={'experiment_id':run.name,'dataset':sample['dataset'],'id':sample['id'],'subject':sample['subject'],'question_type':sample['question_type'],
              'gold':sample['gold'],'response_raw':answer.text,'extracted_answer':e['answer'],'extraction_method':e['method'],'extraction_failure':e['failure'],
              'correct':correct,'parse_failure':not aborted and e['answer'] is None,
              'missing_final_answer':raw_e['answer'] is None and raw_e['failure']=='missing_final_answer',
              'explicit_incorrect':not aborted and e['answer'] is not None and not correct,
              'input_tokens':actual_input,'output_tokens':len(answer.token_ids),'max_new_tokens':8192,
              'generated_token_ids':list(answer.token_ids),'output_cap_reached':len(answer.token_ids)>=8192,'finish_reason':answer.finish_reason,
              'repetition_evidence':getattr(output,'repetition_evidence',None),'elapsed_seconds':time.perf_counter()-start,
              'image_sha256':sample['image_hashes'],'image_grid_thw':grids,'retry':False,'execution_error':None,'evaluation_contract_hash':contract_hash}
            with response_path.open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
            results.append(row);completed.add(row['id'])
            save(run/'progress.json',{'completed_questions':len(results),'planned_questions':len(samples),'optimizer_steps':0,
              'last_id':row['id'],'peak_device_memory_used_mib':monitor.peak,'parent_allocator_is_not_vllm_worker':True})
            print('EVALUATED',len(results),len(samples),row['id'],row['finish_reason'],row['correct'],flush=True)
    except BaseException as e:
        save(run/'logs/evaluation_error.json',{'error':repr(e),'current_id':sample['id'] if 'sample' in locals() else None,
          'completed_questions':len(results),'stage':'see execution_phase.json','completed_responses_preserved':True})
        raise
    finally:
        try:
            if engine is not None:engine.close()
        finally:
            monitor.close()
            save(run/'device_memory_summary.json',{'peak_device_memory_used_mib':monitor.peak,
                'includes_worker_parent_and_display':True,'torch_allocator_peaks_measured':False})
    if len(results)!=len(samples):raise Blocked('Evaluation incomplete; denominator cannot shrink')
    mmmu=[r for r in results if r['dataset']=='MMMU'];fields=['id','subject','question_type','extracted_answer','correct','extraction_method','finish_reason','input_tokens','output_tokens']
    (run/'public_predictions.jsonl').write_text(''.join(json.dumps({k:r[k] for k in fields},ensure_ascii=False)+'\n' for r in mmmu))
    scores={dataset:aggregate([r for r in results if r['dataset']==dataset]) for dataset in ['ChartQA','TabMWP']}
    save(run/'dataset_scores.json',scores)
    return {'MMMU_count':len(mmmu),'generation_count':len(results),'adapter_reload_verified':True,'reload_verified':True,
      'original_model_inference_requests':0,'B0_protocol_sha256':sha(b0run/'evaluation_protocol.json'),'overall':aggregate(mmmu),
      'by_subject':{s:aggregate([r for r in mmmu if r['subject']==s]) for s in sorted({r['subject'] for r in mmmu})},
      'by_question_type':{s:aggregate([r for r in mmmu if r['question_type']==s]) for s in ['multiple-choice','open']},
      'check_scores':scores,'elapsed_seconds':time.perf_counter()-started,'smoke':cfg.get('smoke',False),
      'evaluation_contract_hash':contract_hash,'peak_allocated_bytes':None,'peak_reserved_bytes':None,
      'peak_device_memory_used_mib':monitor.peak,'parent_allocator_is_not_vllm_worker':True,
      'rescoring_delta_pp':None,'interpretation':'smoke pipeline only' if cfg.get('smoke') else 'single-seed development comparison; not replicated improvement evidence'}
