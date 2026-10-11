"""Public aggregates only; all raw source/question/response artifacts stay private."""
import collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    b0run=Path((ROOT/'.restore/b0_run.txt').read_text().strip());b0=json.loads((b0run/'metrics.json').read_text())
    state=json.loads((ROOT/'.restore/suite_state.json').read_text()) if (ROOT/'.restore/suite_state.json').exists() else {'status':'preparing','completed':{}}
    cfg=json.loads((ROOT/'configs/restore/E123.json').read_text());manifest=json.loads((Path(cfg['processed_dir'])/'manifest.json').read_text())
    audit_runs=sorted((ROOT/'runs').glob('*restore_validate_B0'));audit=json.loads((audit_runs[-1]/'metrics.json').read_text()) if audit_runs else None
    output=ROOT/'docs/experiments/RESTORE_E123';output.mkdir(exist_ok=True)
    base={'mixed_historical':{'correct':563,'total':900,'accuracy_pct':100*563/900},'B0_uniform':b0['overall'],
        'B0_subjects':b0['by_subject'],'B0_question_formats':b0['by_question_type'],'rescoring_delta_pp':b0['rescoring_delta_pp'],
        'initial493_not_used':True,'status':state['status'],'adapters':{},'checks_base':None,'processor_audit':audit,
        'previous_train_samples_identical':False,'reviewed_historical_work_type_labels_available':False,
        'non_target_task_change':None,'seed_count':1,'original_MMMU_inference_count':0}
    if 'heldout_base' in state['completed']:base['checks_base']=json.loads((Path(state['completed']['heldout_base'])/'metrics.json').read_text()).get('check_scores')
    for name in ['E1','E2','E3']:
        key=name+'_evaluate'
        if key not in state['completed']:
            base['adapters'][name]={'status':'pending' if not state.get('failed') else 'not_completed','correct':None,'training_updates':0}
            train=state['completed'].get(name+'_train')
            if train:base['adapters'][name]['training_updates']=json.loads((Path(train)/'metrics.json').read_text()).get('optimizer_steps',0)
            continue
        m=json.loads((Path(state['completed'][key])/'metrics.json').read_text())
        value={k:m[k] for k in ['overall','by_subject','by_question_type','check_scores']};value['status']='completed'
        value['delta_from_B0_pp']=m['overall']['correct_pct']-b0['overall']['correct_pct'];base['adapters'][name]=value
    (output/'metrics.json').write_text(json.dumps(base,ensure_ascii=False,indent=2)+'\n')
    public_config={k:cfg[k] for k in ['model_id','model_revision','seed','lora_rank','lora_alpha','lora_dropout','micro_batch','gradient_accumulation','optimizer_steps','warmup_steps','learning_rate','weight_decay','optimizer','scheduler','max_sequence_length','min_pixels','max_pixels','generation_max_new_tokens','eval_max_input_tokens','backend','purpose']}
    public_config['data']={'version':manifest['processing_version'],'sources':manifest['sources'],'files':{n:{'sha256':v['sha256'],'count':v['count']} for n,v in manifest['files'].items()},'previous_sample_identity_verified':False}
    public_config['evaluation']=json.loads((b0run/'evaluation_protocol.json').read_text())['settings']
    (output/'config.json').write_text(json.dumps(public_config,indent=2)+'\n')
    lines=['# Current-host E1/E2/E3 restoration','',f"Suite status: **{state['status']}**.",'',
      'Current host verified by the agent outside the Codex sandbox: user user, AIX408-4, one RTX4090 (24564MiB), driver535.274.02. At probe:0% utilization,18MiB used, no compute processes. Training Python torch2.7.0+cu118 passed CUDA arithmetic and BF16 backward. Codex sandbox nvidia-smi exit9 and missing CUDA access are recorded separately; these are not host driver failures. No driver changes.', '',
      'Separate training venv inherits existing sam3 torch2.7.0+cu118; own transformers4.57.1, PEFT0.17.1, accelerate1.15.0. Existing parent packages preserved. Training torch differs from the historical installation; all three experiments use the same restored environment. Evaluation requires pinned vLLM0.11.2/torch2.9.0/transformers4.57.1/qwen-vl-utils0.0.14; actual engine compatibility is checked before main learning. Model weights use a volatile RAM cache because persistent disk is constrained; adapters/results stay on disk. Reboot requires re-downloading the pinned model.', '',
      'MMMU official validation was recovered at revision98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68:900 questions,982 images. Historical manifest SHA256 eed2e5d5327debaa7602eb67dbcd12cf7e68358c62f464a170bd196bcf420697 matches exactly. QA/options/image ordering and all decoded images verified. Original MMMU model inference is forbidden. Initial493/900 responses were not used.', '',
      'Official pinned ChartQA train_human and validation, TabMWP train/dev and table PNGs, and password-protected PGPS9K ZIP/diagram PNGs were downloaded. E1=ChartQA1000; E2=500 of E1 plus TabMWP500; E3=PGPS9K1000. Checks100/source. Published historical train manifests/IDs were absent, so these are new seed3407 manifests. E2 ChartQA is stratified by new training-question operations and answer forms, not historical MMMU labels. Exact image/table group overlap across training/checks is zero; exact MMMU text/RGB and dHash<=2 candidates excluded. PGPS/Geometry3K official test diagrams are excluded. Unsafe PGPS option mappings are rejected without rounding or guessing. Relations/programs are not input. Semantic contamination is not exhaustively ruled out.', '',
      '| Model/record | Correct /900 | Accuracy | Meaning |','|---|---:|---:|---|',
      '| Historical mixed |563/900|62.56%|Preserved mixed judge/rules|',
      f"| B0 uniform |{b0['overall']['correct']}/900|{b0['overall']['correct_pct']:.2f}%|Saved-response rescoring; no new inference|" ]
    for name,v in base['adapters'].items():
        if v['status']=='completed':lines.append(f"| {name} |{v['overall']['correct']}/900|{v['overall']['correct_pct']:.2f}%|Delta B0 {v['delta_from_B0_pp']:+.2f}pp|")
        else:lines.append(f"| {name} |—|—|{v['status']}; no performance claim|")
    o=b0['overall'];lines+=['',f"B0: parse failures{o['parse_failures']}, length finishes{o['length_finish']}, repetition aborts{o['repetition_aborts']}. Delta from mixed is−8.00pp from scoring policy, not learning. Exact frozen latest_vllm_final_v2 extraction does not see GT. Unextractable/aborted outputs count wrong; length limits retain extractable final answers. Synthetic checks and all900 real response formats were checked; original scorer code is frozen.",'',
      'Training: BF16 language attention/MLP LoRA r16/alpha32/dropout0.05; vision/merger frozen; micro1/accum4; AdamW lr1e-4/weight_decay0.01; warmup13 actual updates then constant;1epoch/1000 exposures/250 updates; max4096/pixels200704–1003520; seed3407. Separate8-question/2-update smoke adapter never continues into E1/E2/E3. Each main run starts original weights independently. Reload generation is a separate process. GPU stages run sequentially and preserve unrelated jobs.', '',
      'Evaluation reuses historical messages, image processing, CoT suffix,8192 output/49152 context,seed3407,.7/.8/top_k20,repetition1.0/presence1.5 and frozen streaming guard. Dynamic BF16 LoRA kernels are a required adapter-specific backend difference. Expanded input-token counts must equal historical counts. Source heldout checks have a separately authorized original-model comparator; no MMMU re-inference. Work-type/non-target metrics remain unavailable without reviewed old ID-level labels; subjects and MC/open formats are reported separately.', '',
      'Full private run records retain command/config/environment/data hashes/source provenance/progress/errors/checkpoints/responses/decisions. Publication contains code, regeneration settings and aggregates only. Publication failure never re-executes learning. One seed and MMMU validation development comparison; no official-score or replicated-improvement claim.', '',
      'See [metrics.json](metrics.json) for subject/format/error rates and [config.json](config.json) for exact settings and data hashes. [Host commands](HOST_RUN.md).']
    if state.get('failed'):
        failure=Path(state['failed_run'])/'metrics.json';m=json.loads(failure.read_text());lines+=['',f"Stopped stage: {state['failed']}. Status: {m.get('status')}. Error: `{m.get('error')}`. Completed work preserved; no dependent evaluation or automatic training restart."]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(output)
if __name__=='__main__':main()
