"""CPU-only recomputation of all frozen preference cache input contracts."""
import hashlib,json,pathlib,sys
from core import *
from run import ROOT,ART,cfgget,adapter
cfg=cfgget();p=processor(cfg);checks=[]
for q in rows(ART/'mine/preference_cache_PRIVATE.jsonl'):
 for letter,saved in q['cache_contracts'].items():
  _,b,P,label_ids=choice_input(p,q['row'],q['prefix'],letter)
  current={'sample_id':q['id'],'checkpoint_sha256':sha(adapter('mixed_sft')/'adapter_model.safetensors'),'processor_hashes':{f:sha(pathlib.Path(cfg['model_path'])/f) for f in ['tokenizer.json','tokenizer_config.json','chat_template.json','preprocessor_config.json']},'template':q['prefix'],'source_image_hashes':[sha(x) for x in q['row']['images']],'input_ids':b['input_ids'][0].tolist(),'completion_mask':[False]*P+[True]*(b['input_ids'].shape[1]-P),'label_token_ids':label_ids,'pixel_values_sha256':hashlib.sha256(b['pixel_values'].numpy().tobytes()).hexdigest(),'image_grid_thw':b['image_grid_thw'].tolist()}
  assert h(current)==saved['key'] and all(current[k]==saved[k] for k in current)
  checks.append({'id':q['id'],'candidate':letter,'contract_matches':True,'key':saved['key']})
save(ROOT/'cache_input_audit.json',{'all_candidates':len(checks),'questions':len({r['id'] for r in checks}),'checks':checks});print('ALL_ACTUAL_CACHE_INPUTS_MATCH',len(checks))
