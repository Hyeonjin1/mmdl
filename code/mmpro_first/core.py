"""Actual Qwen processor, disjoint weighted masks, stable label-token boundaries."""
import hashlib,json,pathlib,re,time,random,os
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,set_seed,StoppingCriteria,StoppingCriteriaList
from peft import LoraConfig,get_peft_model,PeftModel
R=pathlib.Path(os.environ.get('MMPRO_PROJECT','/home/user/mmdl-work'));MODEL=pathlib.Path('/dev/shm/mmdl-user/ebb281ec70b05090aa6165b016eac8ec08e71b17')
INSTRUCTION='''Solve the multiple-choice question using the provided image(s) and question.
Identify the relevant visual evidence, such as labels, units, axes, or spatial
relationships, when applicable. Compare the plausible choices and check any
calculation before choosing exactly one option.

Return exactly two blocks:
<think>Write a concise explanation grounded in the relevant evidence.</think>
<answer>Write only the uppercase letter of the selected option.</answer>'''
OPEN=INSTRUCTION.replace('multiple-choice question','question').replace('Compare the plausible choices and check any\ncalculation before choosing exactly one option.','Check any calculation before giving the final answer.').replace('Write only the uppercase letter of the selected option.','Write only the concise final answer, including necessary units.')
DEFAULT={'seed':42,'model_revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17','model_path':str(MODEL),'dtype':'bfloat16','lora_r':8,'lora_alpha':16,'lora_dropout':0.05,'learning_rate':2e-5,'weight_decay':0.01,'max_grad_norm':1.,'sft_updates':2,'sft_accum':3,'preference_updates':2,'preference_accum':2,'min_pixels':200704,'max_pixels':401408,'max_sequence_length':3072,'generation_max_new_tokens':384,'reasoning_max_new_tokens':512,'beta':0.1,'answer_weight':3,'loss_denominator':5,'gradient_checkpointing':True,'packing':False,'instruction':INSTRUCTION,'target_whitespace':'newline inside answer tag ensures actual label token boundaries','eval_scope':'pipeline_smoke_not_full_accuracy'}
def save(p,x):pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def rows(p):return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]
def write(p,rr):pathlib.Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rr))
def processor(cfg):
 p=AutoProcessor.from_pretrained(cfg['model_path'],local_files_only=True);p.image_processor.size={'shortest_edge':cfg['min_pixels'],'longest_edge':cfg['max_pixels']};return p

def messages(row):
 mc=bool(row.get('options'));content=[{'type':'text','text':INSTRUCTION if mc else OPEN}];images=list(row['images'])
 if row.get('screenshot_only'):
  assert len(images)==1;content.append({'type':'image'});return [{'role':'user','content':content}]
 q=row['question'];matches=list(re.finditer(r'<image\s+(\d+)>',q));provided=[];outimages=[]
 if matches:
  start=0
  for m in matches:
   content.append({'type':'text','text':q[start:m.start()]});idx=int(m[1])-1
   if idx>=len(images):raise ValueError('Question image placeholder has no matching image slot')
   content.append({'type':'image'});outimages.append(images[idx]);provided.append(idx);start=m.end()
  content.append({'type':'text','text':q[start:]})
  for idx,img in enumerate(images):
   if idx not in provided:content.append({'type':'image'});outimages.append(img)
  assert outimages==images,'Preserve official image ordering; unsupported repeated/out-of-order references excluded'
 else:
  content += [{'type':'image'} for x in images];content.append({'type':'text','text':q})
 if mc:content.append({'type':'text','text':'\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(row['options']))})
 return [{'role':'user','content':content}]
def target(row):return '<think>'+row['reason'].strip()+'</think>\n<answer>\n'+str(row['answer']).strip()+'\n</answer>'
def process_text(p,row,text):
 ims=[Image.open(x).convert('RGB') for x in row['images']]
 return p(text=[text],images=ims,return_tensors='pt',padding=False,truncation=False)
def encode(p,row,cfg,answer=None):
 msg=messages(row);prefix=p.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);ans=target(row) if answer is None else answer;full=p.apply_chat_template(msg+[{'role':'assistant','content':ans}],tokenize=False,add_generation_prompt=False)
 a=process_text(p,row,prefix);b=process_text(p,row,full);P=a['input_ids'].shape[1];T=b['input_ids'].shape[1];assert torch.equal(a['input_ids'][0],b['input_ids'][0,:P]),'chat prefix merges across assistant boundary'
 if T>cfg['max_sequence_length']:raise ValueError('sequence exceeds limit; no image/answer truncation')
 assert torch.equal(a['pixel_values'],b['pixel_values']);tok=p.tokenizer(ans,add_special_tokens=False,return_offsets_mapping=True);ids=tok['input_ids'];assert ids==b['input_ids'][0,P:P+len(ids)].tolist(),'assistant token offsets mismatch'
 imageid=p.tokenizer.convert_tokens_to_ids('<|image_pad|>');assert int((b['input_ids']==imageid).sum())>0
 masks={name:torch.zeros(T,dtype=torch.bool) for name in ['reason','answer','format']}
 why_start=ans.index('<think>')+7;why_end=ans.index('</think>');ans_start=ans.index('<answer>')+8;ans_end=ans.index('</answer>');content_start=ans_start+len(ans[ans_start:ans_end])-len(ans[ans_start:ans_end].lstrip());content_end=ans_end-(len(ans[ans_start:ans_end])-len(ans[ans_start:ans_end].rstrip()))
 for i,(s,e) in enumerate(tok['offset_mapping']):
  if why_start<=s and e<=why_end:region='reason'
  elif content_start<=s and e<=content_end:region='answer'
  elif s<content_end and e>content_start:raise ValueError('token crosses answer/tag boundary')
  else:region='format'
  masks[region][P+i]=True
 eos=p.tokenizer.convert_tokens_to_ids('<|im_end|>');stop=P+len(ids);assert b['input_ids'][0,stop].item()==eos;masks['format'][stop]=True
 for v in masks.values():assert not v[:P].any() and not v[b['attention_mask'][0]==0].any() and not v[b['input_ids'][0]==imageid].any()
 assert all(v.any() for v in masks.values());assert not (masks['reason']&masks['answer']).any();assert sum(int(v.sum()) for v in masks.values())==len(ids)+1
 detail={'id':row['id'],'prefix_tokens':P,'total_tokens':T,'region_counts':{k:int(v.sum()) for k,v in masks.items()},'next_token_shift':'logits positions P-1..T-2 predict input_ids[P:T]','token_ids':b['input_ids'][0].tolist(),'region_by_token':['prompt_or_padding' if not any(v[i] for v in masks.values()) else next(k for k,v in masks.items() if v[i]) for i in range(T)],'decoded_input':p.decode(b['input_ids'][0],skip_special_tokens=False),'decoded_target':ans}
 return b,masks,P,detail

def load(cfg,out,adapter=None,train=False):
 set_seed(cfg['seed']);torch.set_num_threads(2);params={'local_files_only':True,'dtype':torch.bfloat16,'device_map':{'':0},'low_cpu_mem_usage':True};receipt={'requested':'flash_attention_2','actual':None}
 # One real attempt, then documented SDPA fallback. No package/driver changes.
 prior=out.parent/'backend.json'
 if prior.exists():receipt=json.loads(prior.read_text());backend=receipt['actual']
 else:
  try:
   base=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_path'],attn_implementation='flash_attention_2',**params);receipt['actual']='flash_attention_2';backend='flash_attention_2'
  except (ImportError,ModuleNotFoundError,ValueError) as e:receipt.update(actual='sdpa',failure=repr(e));backend='sdpa'
  save(prior,receipt)
 if backend!='flash_attention_2' or 'base' not in locals():base=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_path'],attn_implementation=backend,**params)
 if adapter:model=PeftModel.from_pretrained(base,str(adapter),is_trainable=train,autocast_adapter_dtype=False)
 elif train:
  names=[name for name,m in base.named_modules() if isinstance(m,torch.nn.Linear) and 'language_model' in name and ('.self_attn.' in name or '.mlp.' in name)];assert names and not any('visual' in n or 'merger' in n for n in names);model=get_peft_model(base,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],target_modules=names,bias='none',task_type='CAUSAL_LM'),autocast_adapter_dtype=False)
 else:model=base
 if train:
  for n,v in model.named_parameters():
   if v.requires_grad:assert 'lora_' in n and 'language_model' in n and 'visual' not in n and 'merger' not in n;v.data=v.data.to(torch.bfloat16)
  model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
 else:model.eval();model.config.use_cache=True
 save(out/'model_state.json',{'base_revision':cfg['model_revision'],'adapter':str(adapter) if adapter else None,'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad),'LoRA_trainable':train,'vision_merger_frozen':all(not v.requires_grad for n,v in model.named_parameters() if 'visual' in n or 'merger' in n),'actual_backend':backend})
 return model

def gpu(b):return {k:v.to('cuda') for k,v in b.items()}
def weighted(model,b,masks,P,w=3,denom=5):
 T=b['input_ids'].shape[1];out=model(**gpu(b),use_cache=False,logits_to_keep=T-P+1);logits=out.logits[0,:-1].float();labels=b['input_ids'][0,P:].to('cuda');ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none');terms={k:ce[v[P:].to('cuda')].mean() for k,v in masks.items()};return (terms['reason']+w*terms['answer']+terms['format'])/denom,terms

def base_prompt(p,row):return p.apply_chat_template(messages(row),tokenize=False,add_generation_prompt=True)
class EndThink(StoppingCriteria):
 def __init__(self,p,start):self.p=p;self.start=start
 def __call__(self,input_ids,scores,**kwargs):return torch.tensor(['</think>' in self.p.decode(input_ids[0,self.start:],skip_special_tokens=False)],device=input_ids.device)
def generate(model,p,row,cfg,reason_only=False):
 pre=base_prompt(p,row)+('<think>' if reason_only else '');batch=process_text(p,row,pre);n=batch['input_ids'].shape[1]
 if n+cfg['reasoning_max_new_tokens' if reason_only else 'generation_max_new_tokens']>cfg['max_sequence_length']:raise ValueError('generation context overflow; no truncation')
 kw={};
 if reason_only:kw['stopping_criteria']=StoppingCriteriaList([EndThink(p,n)])
 with torch.inference_mode():tokens=model.generate(**gpu(batch),max_new_tokens=cfg['reasoning_max_new_tokens' if reason_only else 'generation_max_new_tokens'],do_sample=False,use_cache=True,**kw)[0,n:]
 txt=p.decode(tokens,skip_special_tokens=False);txt=txt.removesuffix('<|im_end|>').removesuffix('<|endoftext|>').rstrip();capped=len(tokens)>=cfg['reasoning_max_new_tokens' if reason_only else 'generation_max_new_tokens'];return ('<think>'+txt if reason_only else txt),{'output_tokens':len(tokens),'input_tokens':n,'truncated':capped}

def choice_input(p,row,prefix,label):
 pre=base_prompt(p,row)+prefix;pref=process_text(p,row,pre);suffix=label+'\n</answer><|im_end|>';allb=process_text(p,row,pre+suffix);P=pref['input_ids'].shape[1];assert torch.equal(pref['input_ids'][0],allb['input_ids'][0,:P]),'label prefix boundary mismatch';bare=process_text(p,row,pre+label);assert torch.equal(pref['input_ids'][0],bare['input_ids'][0,:P]);labids=bare['input_ids'][0,P:].tolist();assert labids and allb['input_ids'][0,P:P+len(labids)].tolist()==labids
 return pref,allb,P,labids

def suffix_score(model,b,P,label_len=None):
 n=b['input_ids'].shape[1]-P;out=model(**gpu(b),use_cache=False,logits_to_keep=n+1);lp=out.logits[0,:-1].float().log_softmax(-1);picked=lp.gather(-1,b['input_ids'][0,P:].to('cuda').unsqueeze(-1)).squeeze(-1);return picked[:label_len].sum() if label_len is not None else picked.sum()
def choice_scores(model,p,row,prefix):
 labels=[chr(65+i) for i in range(len(row['options']))];prepared=[choice_input(p,row,prefix,l) for l in labels]
 if all(len(z[3])==1 for z in prepared):
  common=prepared[0][0];assert all(torch.equal(z[0]['input_ids'],common['input_ids']) for z in prepared);lp=model(**gpu(common),use_cache=False,logits_to_keep=1).logits[0,-1].float().log_softmax(-1);return {l:lp[z[3][0]] for l,z in zip(labels,prepared)}
 return {l:suffix_score(model,z[1],z[2],len(z[3])) for l,z in zip(labels,prepared)}
def reason_prefix(txt):
 m=re.search(r'^\s*<think>(.*?)</think>',txt,re.S)
 if not m or not m[1].strip():return None
 return '<think>'+m[1]+'</think>\n<answer>\n'
def parse_free(txt,options):
 m=re.fullmatch(r'\s*<think>(.*?)</think>\s*<answer>\s*([A-Z])\s*</answer>\s*',txt,re.S);return m[2] if m and m[1].strip() and m[2] in [chr(65+i) for i in range(len(options))] else None
