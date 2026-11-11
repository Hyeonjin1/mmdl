"""One GPU process at a time: validate -> SFT -> mine -> DPO/rank -> dev -> locked Pro smoke."""
import argparse,ast,collections,fcntl,gc,hashlib,json,os,pathlib,random,subprocess,sys,time,traceback
import torch
from core import *
ROOT=pathlib.Path(os.environ.get('MMPRO_WORK',R/'.restore/mmpro_first_v1'));DATA=pathlib.Path(os.environ.get('MMPRO_DATA',R/'datasets/processed/MMProFirst_20261111'));ART=pathlib.Path(os.environ.get('MMPRO_ART',R/'runs/MMProFirst_20261111'))
def cfgget():return json.loads((ROOT/'config.json').read_text())
def validrows(name):return rows(ROOT/(name+'_validated.jsonl'))
def stage_dir(stage):
 d=ART/stage
 if (d/'complete.json').exists():raise RuntimeError('Stage already complete; preserve it and use new run ID for changes: '+stage)
 if d.exists():d.rename(ART/(stage+'_failed_attempt_'+str(time.time_ns())))
 d.mkdir(parents=True,exist_ok=False);save(d/'config.json',cfgget());(d/'commands.txt').write_text(sys.executable+' '+' '.join(sys.argv)+'\n');return d

def validation(cfg):
 p=processor(cfg);rejected=[];counts={};sample=None
 for name in ['arxiv_train','arxiv_dev','fine_train','openmm_train']:
  accepted=[]
  for row in rows(DATA/(name+'.jsonl')):
   try:
    if row['source']=='MMFineReason' and not row['options']:
     choices=list(re.finditer(r'(?m)^\(?([A-J])[).:]\s*(.*)',row['question']))
     if choices:
      assert [m[1] for m in choices]==[chr(65+i) for i in range(len(choices))], 'MC choice labels not contiguous actual range'
      options=[m[2].strip() for m in choices];assert row['answer'] in [chr(65+i) for i in range(len(options))], 'MC annotation not an actual option key'
      row={**row,'question':row['question'][:choices[0].start()].rstrip().removesuffix('Choices:').strip(),'options':options,'normalization':'actual inline options extracted; original annotation key validated'}
     elif re.fullmatch('[A-J]',str(row['answer'])):raise ValueError('single-letter annotation without verifiable actual options')
    assert row['image_sha256']==[sha(x) for x in row['images']];batch,masks,P,detail=encode(p,row,cfg);accepted.append(row)
    if sample is None:sample=detail;sample['masks']={k:v.tolist() for k,v in masks.items()};assert all(not v for v in sample['masks']['answer'][:P]);assert detail['region_counts']['answer']>0
   except Exception as e:rejected.append({'id':row['id'],'source':row['source'],'error':repr(e)})
  write(ROOT/(name+'_validated.jsonl'),accepted);counts[name]=len(accepted)
 save(ROOT/'mask_example_PRIVATE.json',sample);save(ROOT/'validation.json',{'counts':counts,'validated_file_hashes':{n:sha(ROOT/(n+'_validated.jsonl')) for n in counts},'rejected':rejected,'no_truncation':True,'assistant_only_loss':True,'disjoint_region_masks':True,'next_token_shift_verified':True,'samplewise_region_means':True,'padding':'microbatch1 has no real padding; synthetic4 positions masked'})
 assert all(counts[n]>0 for n in counts),'A required source/dev set has no valid examples'
 # Explicit synthetic padding does not acquire any supervised region.
 for mask in sample['masks'].values():
  padded=torch.tensor(mask+[False]*4);attention=torch.tensor([1]*len(mask)+[0]*4);assert not padded[attention==0].any()
 print('VALIDATED',counts,flush=True)

def sft(cfg,stage,d):
 p=processor(cfg);model=load(cfg,d,train=True);torch.cuda.reset_peak_memory_stats();started=time.time();dev=validrows('arxiv_train');mixed=stage!='arxiv_sft';w=1 if stage=='mixed_w1' else cfg['answer_weight']
 pools=[dev,validrows('fine_train'),validrows('openmm_train')] if mixed else [dev];order=[]
 for step in range(cfg['sft_updates']):
  for micro in range(cfg['sft_accum']):
   pool=pools[micro%len(pools)];idx=(step+(micro//len(pools)))%len(pool);order.append(pool[idx])
 opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);history=[];opt.zero_grad(set_to_none=True)
 for st in range(cfg['sft_updates']):
  vals=[]
  for r in order[st*cfg['sft_accum']:(st+1)*cfg['sft_accum']]:
   b,m,P,detail=encode(p,r,cfg);loss,t=weighted(model,b,m,P,w,cfg['loss_denominator']);assert torch.isfinite(loss);(loss/cfg['sft_accum']).backward();vals.append({'id':r['id'],'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in t.items()},'region_counts':detail['region_counts']})
  norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],cfg['max_grad_norm'],error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);record={'update':st+1,'micro':vals,'grad_norm':float(norm)};history.append(record);save(d/'progress.json',record);print('UPDATE',stage,st+1,vals,flush=True)
 model.save_pretrained(d/'adapter',safe_serialization=True);p.save_pretrained(d/'adapter');write(d/'losses.jsonl',history)
 metrics={'optimizer_steps':len(history),'exposures':len(order),'unique_questions':len({r['id'] for r in order}),'source_exposures':dict(collections.Counter(r['source'] for r in order)),'ordered_ids':[r['id'] for r in order],'answer_weight':w,'loss_denominator':cfg['loss_denominator'],'seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'adapter_hashes':{f.name:sha(f) for f in (d/'adapter').iterdir() if f.is_file()}};save(d/'metrics.json',metrics);return metrics

def preflight(cfg,d):
 p=processor(cfg);model=load(cfg,d,train=True);row=validrows('arxiv_train')[0];b,m,P,detail=encode(p,row,cfg);torch.cuda.reset_peak_memory_stats();loss,terms=weighted(model,b,m,P);loss.backward();assert any(v.grad is not None and bool(torch.count_nonzero(v.grad)) for n,v in model.named_parameters() if 'lora_' in n);assert all(v.grad is None for n,v in model.named_parameters() if 'visual' in n or 'merger' in n)
 save(d/'mask_example_PRIVATE.json',detail);return {'image_forward_backward_passed':True,'loss':float(loss.detach()),'region_loss':{k:float(v.detach()) for k,v in terms.items()},'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'optimizer_steps':0,'preflight_adapter_not_reused':True}

def adapter(stage):return ART/stage/'adapter'
def mining(cfg,d):
 p=processor(cfg);model=load(cfg,d,adapter('mixed_sft'));out=[];fail=[];started=time.time();weightsha=sha(adapter('mixed_sft')/'adapter_model.safetensors');procsha={f:sha(pathlib.Path(cfg['model_path'])/f) for f in ['tokenizer.json','tokenizer_config.json','chat_template.json','preprocessor_config.json']}
 for row in validrows('arxiv_train')[:32]:
  try:
   txt,meta=generate(model,p,row,cfg,True);pre=reason_prefix(txt)
   if pre is None or meta['truncated']:fail.append({'id':row['id'],'reason':'missing_complete_generated_reasoning_or_length','generation':txt,'meta':meta});continue
   with torch.no_grad():scores={k:float(v) for k,v in choice_scores(model,p,row,pre).items()};wrong=sorted([k for k in scores if k!=row['answer']],key=lambda k:(-scores[k],k))[:2];candidates=[row['answer']]+wrong;refs={};contracts={}
   for letter in candidates:
    pref,b,P,lab=choice_input(p,row,pre,letter)
    with torch.no_grad():refs[letter]=float(suffix_score(model,b,P))
    contract={'sample_id':row['id'],'checkpoint_sha256':weightsha,'processor_hashes':procsha,'template':pre,'source_image_hashes':row['image_sha256'],'input_ids':b['input_ids'][0].tolist(),'completion_mask':[False]*P+[True]*(b['input_ids'].shape[1]-P),'label_token_ids':lab,'pixel_values_sha256':hashlib.sha256(b['pixel_values'].numpy().tobytes()).hexdigest(),'image_grid_thw':b['image_grid_thw'].tolist()};contracts[letter]={'key':h(contract),**contract}
   out.append({'id':row['id'],'row':row,'generated_reasoning':txt,'prefix':pre,'gold':row['answer'],'wrong':wrong,'all_label_logp_sums':scores,'reference_completion_logp_sums':refs,'cache_contracts':contracts,'generation':meta});print('MINED',row['id'],scores,wrong,flush=True)
   if len(out)>=cfg.get('mining_questions',4):break
  except Exception as e:fail.append({'id':row['id'],'reason':repr(e)})
 write(d/'preference_cache_PRIVATE.jsonl',out);save(d/'failures.json',fail);assert out,'No valid generated reasoning / exact token-boundary pairs; no fake negatives'
 # Recompute real completion sums, independent forwards; do not use rounded label proxy.
 checks=[]
 for q in out[:2]:
  for letter,ref in q['reference_completion_logp_sums'].items():
   _,b,P,lab=choice_input(p,q['row'],q['prefix'],letter)
   with torch.no_grad():actual=float(suffix_score(model,b,P))
   assert abs(actual-ref)<1e-4;checks.append({'id':q['id'],'label':letter,'absolute_error':abs(actual-ref),'cache_key':q['cache_contracts'][letter]['key']})
 save(d/'cache_recompute.json',checks);assert sha(adapter('mixed_sft')/'adapter_model.safetensors')==weightsha
 return {'valid_questions':len(out),'pairs':sum(len(q['wrong']) for q in out),'failed_questions':len(fail),'seconds':time.time()-started,'reference_frozen_sha256':weightsha,'cache_recomputed_pairs':len(checks),'reference_scope':'entire completion suffix including close tag and EOS; ranking label tokens only'}

def preference(cfg,stage,d):
 p=processor(cfg);cache=rows(ART/'mine'/'preference_cache_PRIVATE.jsonl');model=load(cfg,d,adapter('mixed_sft'),True);model.eval();validated=[]
 for q in cache[:2]:
  for letter,ref in q['reference_completion_logp_sums'].items():
   _,b,P,lab=choice_input(p,q['row'],q['prefix'],letter);saved=q['cache_contracts'][letter];now={k:v for k,v in saved.items() if k!='key'};assert now['checkpoint_sha256']==sha(adapter('mixed_sft')/'adapter_model.safetensors');assert b['input_ids'][0].tolist()==saved['input_ids'] and h(now)==saved['key'];assert sha(q['row']['images'][0])==q['row']['image_sha256'][0]
   with torch.no_grad():actual=float(suffix_score(model,b,P))
   assert abs(actual-ref)<1e-4;validated.append({'id':q['id'],'label':letter,'absolute_error':abs(actual-ref)})
 save(d/'actual_training_input_cache_checks.json',validated);model.train();opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);limit=min(len(cache),cfg['preference_updates']*cfg['preference_accum']);questions=cache[:limit];hist=[];opt.zero_grad(set_to_none=True);started=time.time();torch.cuda.reset_peak_memory_stats()
 for begin in range(0,len(questions),cfg['preference_accum']):
  batch=questions[begin:begin+cfg['preference_accum']];metrics=[]
  for q in batch:
   if stage=='dpo':
    values={}
    for letter in [q['gold']]+q['wrong']:
     _,b,P,lab=choice_input(p,q['row'],q['prefix'],letter);values[letter]=suffix_score(model,b,P)
    pref=[-torch.nn.functional.logsigmoid(cfg['beta']*((values[q['gold']]-values[w])-(q['reference_completion_logp_sums'][q['gold']]-q['reference_completion_logp_sums'][w]))) for w in q['wrong']];loss=torch.stack(pref).mean()
   else:
    scores=choice_scores(model,p,q['row'],q['prefix']);z=torch.stack([scores[x] for x in [q['gold']]+q['wrong']]);loss=torch.logsumexp(z,0)-z[0]
   assert torch.isfinite(loss);(loss/len(batch)).backward();metrics.append({'id':q['id'],'loss':float(loss.detach()),'negative_pairs':len(q['wrong'])})
  norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],cfg['max_grad_norm'],error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);hist.append({'update':len(hist)+1,'questions':metrics,'grad_norm':float(norm)});save(d/'progress.json',hist[-1]);print('PREFERENCE_UPDATE',stage,hist[-1],flush=True)
 model.save_pretrained(d/'adapter',safe_serialization=True);p.save_pretrained(d/'adapter');write(d/'losses.jsonl',hist);return {'optimizer_steps':len(hist),'exposures':len(questions),'unique_questions':len({q['id'] for q in questions}),'ordered_ids':[q['id'] for q in questions],'per_question_weight':'mean over negatives first for DPO; one rank loss per question','reference_sha256':sha(adapter('mixed_sft')/'adapter_model.safetensors'),'seconds':time.time()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}

def official_parser():
 source=(ROOT/'evaluate.py').read_text();tree=ast.parse(source);names={'parse_multi_choice_response','get_multi_choice_info','check_is_number','normalize_str'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];scope={'re':re,'random':random.Random(42),'np':__import__('numpy'),'ast':ast};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'evaluate.py'),'exec'),scope);return scope

def evaluate(cfg,modelname,d,pro=False):
 p=processor(cfg);model=load(cfg,d,None if modelname=='initial' else adapter(modelname));source=rows(cfg['evaluation_file']) if cfg.get('evaluation_file') else rows(ROOT/'evaluation_selection.jsonl') if pro else validrows('arxiv_dev');out=[];official=official_parser();started=time.time()
 for row in source:
  item={'id':row['id'],'source':row['source'],'model':modelname,'gold':row['answer'],'options_count':len(row['options'])}
  try:
   txt,meta=generate(model,p,row,cfg);ans=parse_free(txt,row['options']);item.update(response_raw=txt,free_prediction=ans,format_ok=ans is not None,free_correct=ans==row['answer'] and not meta['truncated'],free_generation=meta)
   if pro:
    all_choices=[chr(65+i) for i in range(len(row['options']))];guess=official['parse_multi_choice_response'](txt,all_choices,{k:v for k,v in zip(all_choices,row['options'])});item.update(official_prediction=guess,official_correct=guess==row['answer'],official_parser_may_guess_on_failure=ans is None)
   reason,rm=generate(model,p,row,cfg,True);prefix=reason_prefix(reason);item.update(reasoning_generation_raw=reason,reasoning_generation=rm)
   if prefix is None or rm['truncated']:item.update(ll_prediction=None,ll_correct=False,ll_failure='missing_complete_generated_reasoning')
   else:
    with torch.no_grad():scores={k:float(v) for k,v in choice_scores(model,p,row,prefix).items()}
    pred=max(scores,key=scores.get);item.update(ll_prediction=pred,ll_correct=pred==row['answer'],label_logp_sums=scores,gold_minus_best_wrong=scores[row['answer']]-max(v for k,v in scores.items() if k!=row['answer']))
  except Exception as e:item.update(error=repr(e),free_correct=False,ll_correct=False,format_ok=False)
  out.append(item);write(d/'predictions_PRIVATE.jsonl',out);print('EVAL',modelname,row['source'],row['id'],item.get('free_correct'),item.get('ll_correct'),flush=True)
 metrics={}
 for name in sorted({r['source'] for r in out}):
  rr=[r for r in out if r['source']==name];margins=[r['gold_minus_best_wrong'] for r in rr if r.get('gold_minus_best_wrong') is not None];metrics[name]={'n':len(rr),'free_correct':sum(r['free_correct'] for r in rr),'ll_correct':sum(r['ll_correct'] for r in rr),'format_ok':sum(r['format_ok'] for r in rr),'free_truncated':sum(r.get('free_generation',{}).get('truncated',False) for r in rr),'reasoning_truncated':sum(r.get('reasoning_generation',{}).get('truncated',False) for r in rr),'errors':sum('error' in r for r in rr),'mean_gold_minus_wrong':sum(margins)/len(margins) if margins else None,'official_correct':sum(r.get('official_correct',False) for r in rr) if pro else None}
 return {'by_source':metrics,'seconds':time.time()-started,'reload_and_generation_checked':True,'official_parser_source_commit':'268471d0d488258990025331c7528359c324aa25' if pro else None,'official_prompt_not_reproduced':'user fixed two-tag instruction used' if pro else None,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['validate','preflight','arxiv_sft','mixed_sft','mixed_w1','mine','dpo','ranking','evaluate','pro']);ap.add_argument('--model',default='initial');ap.add_argument('--evaluation-file');ap.add_argument('--evaluation-name');a=ap.parse_args();cfg=cfgget();cfg.update({'evaluation_file':a.evaluation_file} if a.evaluation_file else {});key=a.stage+('_'+a.model if a.stage in ['evaluate','pro'] else '')+('_'+a.evaluation_name if a.evaluation_name else '');lock=(ROOT/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.stage=='validate':validation(cfg);return
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU compute job already running; do not duplicate/terminate'
 d=stage_dir(key);start=time.time()
 try:
  if a.stage=='preflight':m=preflight(cfg,d)
  elif a.stage in ['arxiv_sft','mixed_sft','mixed_w1']:m=sft(cfg,a.stage,d)
  elif a.stage=='mine':m=mining(cfg,d)
  elif a.stage in ['dpo','ranking']:m=preference(cfg,a.stage,d)
  else:m=evaluate(cfg,a.model,d,a.stage=='pro')
  save(d/'metrics.json',m);save(d/'complete.json',{'status':'completed','metrics':m,'code_sha256':{f.name:sha(f) for f in pathlib.Path(__file__).parent.glob('*.py')},'manifest_sha256':sha(DATA/'manifest.json') if (DATA/'manifest.json').exists() else None,'wall_seconds':time.time()-start})
 except Exception:
  (d/'failure.log').write_text(traceback.format_exc());save(d/'failed.json',{'status':'failed','error':traceback.format_exc(),'wall_seconds':time.time()-start});raise
if __name__=='__main__':main()
