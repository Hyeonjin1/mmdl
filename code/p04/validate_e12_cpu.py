"""Validate acquired E1/E2 inputs/assistant masks independently of E3/GPU access."""
import json,re,sys,time,traceback
from pathlib import Path
from records import ROOT,save,sha,index,now
from host_steps import create
from rescore_latest import SNAP,read

def main():
    run=create('validate_E1_E2_cpu');start=time.perf_counter()
    root=ROOT/'runs/P04_20261010T154918Z_prepare_B0/partial_processed'
    cfg=json.loads((ROOT/'configs/P04/common.json').read_text());cfg.update(stage='validate_E1_E2_cpu',processed_dir=str(root),scheduler='linear warmup 13 updates then constant',
      purpose='E1/E2 acquired-data CPU processor and assistant-loss validation; E3 independent')
    save(run/'config_resolved.json',cfg)
    (run/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/user/MMDL\n.venv-p03/bin/python code/p04/validate_e12_cpu.py\n')
    try:
        from learning import encode,processor_for
        from PIL import Image,ImageDraw
        import torch
        torch.set_num_threads(1)
        manifest=json.loads((root/'manifest.json').read_text());allrows={}
        for name in ['E1','E2','ChartQA_check','TabMWP_check']:
            path=root/(name+'.jsonl');assert sha(path)==manifest['files'][name]['sha256'];allrows[name]=read(path)
        assert len(allrows['E1'])==len(allrows['E2'])==1000
        assert len(allrows['ChartQA_check'])==len(allrows['TabMWP_check'])==100
        assert sum(r['source']=='ChartQA' for r in allrows['E2'])==500
        assert sum(r['source']=='TabMWP' for r in allrows['E2'])==500
        assert {r['id'] for r in allrows['E2'] if r['source']=='ChartQA'}.issubset({r['id'] for r in allrows['E1']})
        training=allrows['E1']+allrows['E2'];held=allrows['ChartQA_check']+allrows['TabMWP_check']
        assert not {r['group_id'] for r in training}&{r['group_id'] for r in held}
        assert not {r['pixel_sha256'] for r in training}&{r['pixel_sha256'] for r in held}
        norm=lambda q:re.sub(r'\s+',' ',q).strip().casefold()
        evalrows=read(SNAP/'results/validation900_cot8192_guard_full/predictions.jsonl')
        evalquestions={norm(r['annotation']['question']) for r in evalrows}
        qoverlap=[r['id'] for r in training if norm(r['question']) in evalquestions]
        assert not qoverlap,'Confirmed exact question overlap: '+str(qoverlap)
        unique={r['id']:r for r in training+held};processor=processor_for(cfg);summaries=[];samples=[];selected=[]
        save(run/'data_manifest.json',{'processing_manifest_sha256':sha(root/'manifest.json'),
             'files':{n:manifest['files'][n] for n in allrows},'unique_rows_checked':len(unique),
             'MMMU_recent_question_overlap_count':0,'MMMU_image_overlap_verified':False,
             'E3_required':False,'train_heldout_groups_disjoint':True})
        save(run/'metrics.json',{'status':'running','optimizer_steps':0,'rows_checked':0})
        with (run/'logs/processor.log').open('w',buffering=1) as log:
            for i,row in enumerate(unique.values()):
                item=dict(row,original_id=row['id'],chart_name=Path(row['image']).name)
                batch,detail=encode(item,processor,cfg)
                assert torch.all(batch['labels'][batch['attention_mask']==0]==-100)
                summaries.append({'id':row['id'],'source':row['source'],'total_tokens':detail['total_tokens'],
                                  'supervised_tokens':detail['supervised_tokens'],'image_tokens':detail['image_tokens'],'image_grid_thw':detail['image_grid_thw']})
                if sum(s['source']==row['source'] for s in selected)<2:
                    samples.append(detail);selected.append(row)
                del batch
                if (i+1)%25==0:
                    save(run/'metrics.json',{'status':'running','optimizer_steps':0,'rows_checked':i+1,'wall_seconds':time.perf_counter()-start})
                    log.write(f'{i+1}/{len(unique)} checked\n')
        save(run/'mask_samples.json',samples);save(run/'token_checks.json',summaries)
        # Gallery contains only local images; not exported publicly.
        sheet=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(selected):
            x=(i%2)*600;y=(i//2)*500
            with Image.open(row['image']) as image:
                image=image.convert('RGB');image.thumbnail((580,350));sheet.paste(image,(x,y))
            draw.text((x+10,y+355),row['id'],fill='black');draw.text((x+10,y+380),row['question'][:90],fill='black');draw.text((x+10,y+410),'Target: '+row['answer'],fill='black')
        sheet.save(run/'sample_review.png')
        metrics={'status':'completed','optimizer_steps':0,'unique_rows_checked':len(unique),
          'E1_rows':1000,'E2_rows':1000,'heldout_rows':200,'max_total_tokens':max(r['total_tokens'] for r in summaries),
          'min_supervised_tokens':min(r['supervised_tokens'] for r in summaries),
          'all_assistant_masks_verified':True,'truncation':False,'real_padding':False,
          'gpu_forward_backward_verified':False,'MMMU_image_overlap_verified':False,
          'E3_required':False,'ready_for_training':False,
          'remaining_prerequisites':['host GPU admission','recent MMMU image identity and image overlap','same-version vLLM adapter evaluation environment'],
          'wall_seconds':time.perf_counter()-start,'ended_utc':now()}
        save(run/'metrics.json',metrics)
        (run/'experiment.md').write_text(f'''# E1/E2 CPU validation — {run.name}

Purpose: independently validate acquired ChartQA/TabMWP inputs, unique QA/groups and assistant-only loss. PGPS9K/E3 is not a prerequisite.

Actual conditions: pinned original Qwen processor; training pixels200704/1003520 and max total4096 from successful P03. E1=1000 ChartQA; E2=500 of E1+500 TabMWP; separate checks100+100. Same original question/short answer SFT; no solutions/tables/annotations in user input. No teacher/KD. No training/inference.

Results: {len(unique)} unique QA processor/mask checks passed; max total{metrics['max_total_tokens']}; min supervised{metrics['min_supervised_tokens']}; no truncation or real padding. Token summaries and at least2/source decoded mask samples saved. Train/check pixel/group intersections and normalized question intersections with recent MMMU are empty. Actual MMMU image overlap remains unverified; semantic duplication not exhaustively checked.

Limits: image gallery requires direct review; no claim of reasoning correctness or GPU forward/backward. Missing E3 does not block E1/E2. Host GPU and exact recent-MMMU images/evaluation environment remain required before dependent experiments. No completed E1/E2 adapter found. Automatic follow-on training is disabled after failed prerequisites.

Next: inspect sample_review.png; verify host GPU and acquire matching MMMU images; then run only incomplete approved E1/E2 independently from original model, with same frozen vLLM evaluation/scorer contract as saved B0. Original model re-inference is disallowed.
''')
    except BaseException as e:
        save(run/'metrics.json',{'status':'error','error':repr(e),'optimizer_steps':0,'wall_seconds':time.perf_counter()-start});(run/'logs/error.log').write_text(traceback.format_exc())
        with (run/'experiment.md').open('a') as f:f.write('\nError: '+repr(e)+'\nDependent execution stopped.\n')
        index();raise
    index();print(run);print(json.dumps(metrics))

if __name__=='__main__':main()
