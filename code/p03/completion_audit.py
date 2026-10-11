"""Verify saved GPU run artifacts without training or model generation."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
from common import ROOT, REQUIRED, read_manifest, save, sha

def audit(cfg,run):
    manifest=read_manifest(cfg,run)
    train=ROOT/'runs'/cfg['completion_train_run']
    evaluation=ROOT/'runs'/cfg['completion_evaluate_run']
    cleanup=ROOT/'runs'/cfg['completion_cleanup_run']
    for source in (train,evaluation,cleanup):
        assert all((source/name).is_file() for name in REQUIRED)
    m=json.loads((train/'metrics.json').read_text())
    e=json.loads((evaluation/'metrics.json').read_text())
    assert m['status']=='completed' and m['optimizer_steps']==20 and m['adapter_saved']
    assert e['status']=='completed' and e['reload_verified'] and e['generation_count']==32
    steps=[json.loads(line) for line in (train/'logs/train_steps.jsonl').read_text().splitlines()]
    assert [s['optimizer_step'] for s in steps]==list(range(1,21))
    assert all(math.isfinite(s['mean_micro_loss']) and all(math.isfinite(x) for x in s['micro_losses']) for s in steps)
    assert sum(s['micro_batches'] for s in steps)==80
    checkpoint=json.loads((train/'checkpoint_manifest.json').read_text())
    adapter=Path(checkpoint['path'])
    assert checkpoint['data_manifest_sha256']==sha(train/'data_manifest.json')==sha(run/'data_manifest.json')
    for name,expected in checkpoint['files_sha256'].items(): assert sha(adapter/name)==expected
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    dtypes=Counter();count=0;changed=False
    with safe_open(adapter/'adapter_model.safetensors',framework='pt',device='cpu') as f:
        for name in f.keys():
            tensor=f.get_tensor(name)
            assert 'lora_' in name and torch.isfinite(tensor).all()
            count+=tensor.numel();dtypes[str(tensor.dtype)]+=1
            if 'lora_B' in name and bool(torch.count_nonzero(tensor)): changed=True
    assert changed and count==m['trainable_parameter_count'] and set(dtypes)=={'torch.bfloat16'}
    rows=[json.loads(line) for line in (evaluation/'responses.jsonl').read_text().splitlines()]
    check={row['original_id']:row for row in manifest['check']}
    summary={}
    for mode in ('base','adapter'):
        subset=[row for row in rows if row['mode']==mode]
        assert len(subset)==16 and len({r['original_id'] for r in subset})==16
        assert set(r['original_id'] for r in subset)==set(check)
        for row in subset:
            source=check[row['original_id']]
            assert row['question']==source['question'] and row['gold']==source['answer']
            assert row['image_sha256']==source['image_sha256']
            assert row['correct']==(row['response_raw'].strip().casefold()==source['answer'].casefold())
            assert row['generation']=={'do_sample':False,'num_beams':1,'max_new_tokens':64}
        correct=sum(r['correct'] for r in subset)
        assert e['scores'][mode]==dict(correct=correct,total=16)
        summary[mode]=dict(correct=correct,total=16,output_cap_reached=sum(r['output_cap_reached'] for r in subset),
            parse_failure=sum(r['parse_failure'] for r in subset),empty_response=sum(not r['response_raw'].strip() for r in subset))
    exposure=json.loads((train/'exposures.json').read_text())
    assert exposure['total']==80 and exposure['unique']==64 and exposure['exposure_histogram']=={'1':48,'2':16}
    with (run/'training_steps.csv').open('x',newline='') as f:
        columns=['optimizer_step','mean_micro_loss','learning_rate','question_exposures','peak_allocated_bytes','peak_reserved_bytes','elapsed_seconds']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:s[k] for k in columns} for s in steps)
    with (run/'generation_checks.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['mode',*summary['base']]);writer.writeheader()
        writer.writerows(dict(mode=mode,**values) for mode,values in summary.items())
    result=dict(verified_optimizer_steps=20,checkpoint_hashes_verified=True,adapter_tensor_count=sum(dtypes.values()),
        adapter_dtypes=dict(dtypes),adapter_parameters=count,adapter_B_nonzero=True,
        generation_checks=summary,question_exposures=80,epochs_equivalent=1.25,
        peak_allocated_gib=m['peak_allocated_bytes']/2**30,peak_reserved_gib=m['peak_reserved_bytes']/2**30,
        training_seconds=m['training_seconds'],first_loss=m['first_loss'],last_loss=m['last_loss'],
        checkpoint_path=str(adapter),reload_verified=True,
        limitations='Saved host GPU execution verified from artifacts; no independent GPU re-execution in agent session. Cap reached does not prove missing final answer. No improvement claim.')
    save(run/'completion_verified.json',result)
    return result
