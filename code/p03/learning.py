"""Transformers + PEFT P03 pipeline; data/processor checks can run on CPU."""
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import random
import time
from common import Blocked, read_manifest, require_gpu, save, sha

def snapshot_path(cfg):
    if cfg.get('model_path'):
        path=Path(cfg['model_path'])
        if not (path/'config.json').is_file():raise Blocked('Explicit pinned model directory missing config.json')
        manifest=path/'processor_manifest.json'
        if not manifest.is_file() or json.loads(manifest.read_text())['revision']!=cfg['model_revision']:
            raise Blocked('Explicit model/processor revision manifest missing or mismatched')
        return str(path)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=cfg["model_id"],revision=cfg["model_revision"],local_files_only=True)

def processor_for(cfg):
    from transformers import AutoProcessor
    p=AutoProcessor.from_pretrained(snapshot_path(cfg),local_files_only=True)
    p.image_processor.size={"shortest_edge":cfg["min_pixels"],"longest_edge":cfg["max_pixels"]}
    assert p.image_processor.size["shortest_edge"]==cfg["min_pixels"]
    assert p.image_processor.size["longest_edge"]==cfg["max_pixels"]
    return p

def user_messages(row):
    # Gold exists only in the separately added assistant message.
    return [{"role":"user","content":[{"type":"image"},{"type":"text","text":row["question"]}]}]

def encode(row,processor,cfg):
    import torch
    from PIL import Image
    if sha(row["image"])!=row["image_sha256"]: raise ValueError("Image hash changed: "+row["image"])
    with Image.open(row["image"]) as im: image=im.convert("RGB")
    user=user_messages(row)
    prefix=processor.apply_chat_template(user,tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(user+[{"role":"assistant","content":row["answer"]}],
                                       tokenize=False,add_generation_prompt=False)
    prefix_inputs=processor(text=[prefix],images=[image],return_tensors="pt",padding=False,truncation=False)
    inputs=processor(text=[full],images=[image],return_tensors="pt",padding=False,truncation=False)
    plen=prefix_inputs["input_ids"].shape[1]; total=inputs["input_ids"].shape[1]
    if not torch.equal(inputs["input_ids"][0,:plen],prefix_inputs["input_ids"][0]):
        raise ValueError("Template/tokenization boundary mismatch")
    if total>cfg["max_sequence_length"]: raise ValueError("No truncation allowed: "+str(total))
    if not torch.equal(inputs["pixel_values"],prefix_inputs["pixel_values"]):
        raise ValueError("Image changed between question and target")
    image_id=processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    eos=processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if "pixel_values" not in inputs or inputs["pixel_values"].numel()==0: raise ValueError("Image was omitted")
    if int((inputs["input_ids"]==image_id).sum())==0: raise ValueError("Missing image tokens")
    tail=inputs["input_ids"][0,plen:]
    ends=(tail==eos).nonzero(as_tuple=False).flatten()
    if ends.numel()!=1: raise ValueError("Expected exactly one assistant end token")
    stop=plen+int(ends[0])+1
    if stop<=plen+1: raise ValueError("No answer tokens remain")
    decoded_answer=processor.decode(inputs["input_ids"][0,plen:stop-1],skip_special_tokens=True)
    if decoded_answer!=row["answer"]: raise ValueError("Assistant answer changed in tokenization")
    labels=torch.full_like(inputs["input_ids"],-100)
    labels[:,plen:stop]=inputs["input_ids"][:,plen:stop]
    labels[inputs["attention_mask"]==0]=-100
    if not torch.all(labels[:,:plen]==-100): raise ValueError("User tokens receive loss")
    if not torch.all(labels[inputs["input_ids"]==image_id]==-100): raise ValueError("Image tokens receive loss")
    if not torch.all(labels[:,stop:]==-100): raise ValueError("Template suffix receives loss")
    inputs["labels"]=labels
    return inputs,dict(original_id=row["original_id"],chart_name=row["chart_name"],
        image_sha256=row["image_sha256"],input_prefix_decoded=processor.decode(inputs["input_ids"][0,:plen],skip_special_tokens=False),
        target_decoded=processor.decode(inputs["input_ids"][0,plen:stop],skip_special_tokens=False),
        source_answer=row["answer"],prefix_tokens=plen,total_tokens=total,
        supervised_tokens=int((labels!=-100).sum()),image_tokens=int((inputs["input_ids"]==image_id).sum()),
        image_grid_thw=inputs["image_grid_thw"].tolist(),
        prefix_no_gold_in_user_message=True,processor_size=dict(processor.image_processor.size),
        loss_mask=(labels[0]!=-100).tolist(),token_ids=inputs["input_ids"][0].tolist())

def all_rows(manifest):
    train=manifest["train"];check=manifest["check"]
    if len(train)!=64 or len(check)!=16: raise ValueError("Wrong split sizes")
    if {r["group_id"] for r in train}&{r["group_id"] for r in check}: raise ValueError("Chart overlap")
    return train+check

def validate(cfg,run):
    import torch
    from PIL import Image,ImageDraw
    manifest=read_manifest(cfg,run);rows=all_rows(manifest)
    p=processor_for(cfg);details=[];summary=[]
    for i,row in enumerate(rows):
        inputs,detail=encode(row,p,cfg)
        # Explicit padding-mask check; actual micro-batch1 has no padding.
        padded_ids=torch.nn.functional.pad(inputs["input_ids"],(0,4),value=p.tokenizer.pad_token_id or 0)
        padded_attention=torch.nn.functional.pad(inputs["attention_mask"],(0,4),value=0)
        padded_labels=torch.nn.functional.pad(inputs["labels"],(0,4),value=-100)
        assert torch.all(padded_labels[padded_attention==0]==-100)
        assert padded_ids.shape==padded_labels.shape
        if i<2: details.append(detail)
        summary.append({k:detail[k] for k in ["original_id","total_tokens","supervised_tokens","image_tokens","image_grid_thw"]})
        print("MASK_CHECK",i,row["chart_name"],detail["total_tokens"],detail["supervised_tokens"],flush=True)
        del inputs
    save(run/"mask_samples.json",details)
    save(run/"validation_checks.json",dict(status="processor_mask_passed_cpu",rows_checked=80,
        checks=summary,no_truncation=True,user_and_image_loss_masked=True,
        padding_test="four synthetic padding positions masked; real microbatch1 unpadded",
        assistant_answer_and_im_end_supervised=True,
        no_gpu_forward_backward_verified=True,processed_manifest_sha256=sha(Path(cfg["processed_dir"])/"data_manifest.json")))
    sheet=Image.new("RGB",(1000,800),"white"); draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows[:2]):
        with Image.open(row["image"]) as im:
            thumb=im.convert("RGB");thumb.thumbnail((480,620));sheet.paste(thumb,(i*500,0))
        draw.text((i*500+10,640),row["chart_name"],fill="black")
        draw.text((i*500+10,675),row["question"][:75],fill="black")
        draw.text((i*500+10,710),"Target: "+row["answer"],fill="black")
    sheet.save(run/"processor_sample_images.png")
    return dict(rows_checked=80,max_total_tokens=max(r["total_tokens"] for r in summary),
                min_supervised_tokens=min(r["supervised_tokens"] for r in summary),
                mask_check="passed on CPU processor; GPU forward/backward not yet verified")

def attach_lora(model,cfg,run,meta=False):
    import torch
    from peft import LoraConfig,get_peft_model,TaskType
    # Discover the actual text backbone, then use full names of its actual Linear objects.
    language=getattr(getattr(model,"model",None),"language_model",None)
    if language is None: raise ValueError("Actual Qwen language_model backbone not found")
    names={id(m):name for name,m in model.named_modules()}
    targets=[]
    for relative,module in language.named_modules():
        if isinstance(module,torch.nn.Linear) and (".self_attn." in "."+relative or ".mlp." in "."+relative):
            full=names[id(module)]
            if any(x in full for x in ["visual","merger","vision"]): raise ValueError("Vision target")
            targets.append(dict(name=full,in_features=module.in_features,out_features=module.out_features))
    if not targets: raise ValueError("No actual language attention/MLP Linear modules")
    for param in model.parameters(): param.requires_grad_(False)
    adapter=get_peft_model(model,LoraConfig(r=cfg["lora_rank"],lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],bias="none",target_modules=[t["name"] for t in targets],
        task_type=TaskType.CAUSAL_LM),autocast_adapter_dtype=False)
    trainable=[]
    for name,param in adapter.named_parameters():
        if param.requires_grad:
            if "lora_" not in name or "language_model" not in name or any(x in name for x in ["visual","merger"]):
                raise ValueError("Unexpected trainable parameter: "+name)
            param.data=param.data.to(dtype=torch.bfloat16)
            trainable.append(dict(name=name,count=param.numel(),dtype=str(param.dtype),device=str(param.device)))
    data=dict(inspected_actual_modules=targets,target_count=len(targets),
              trainable_parameters=trainable,trainable_parameter_count=sum(t["count"] for t in trainable),
              total_parameter_count=sum(p.numel() for p in adapter.parameters()),
              vision_encoder_and_merger_frozen=True,meta_inspection_only=meta)
    save(run/"lora_modules.json",data)
    for target in targets:
        print("LORA_TARGET",target["name"],target["in_features"],target["out_features"],flush=True)
    print(json.dumps({k:v for k,v in data.items() if k not in ["trainable_parameters","inspected_actual_modules"]},indent=2),flush=True)
    return adapter,data

def structure(cfg,run):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig,Qwen3VLForConditionalGeneration,set_seed
    set_seed(cfg["seed"])
    config=AutoConfig.from_pretrained(snapshot_path(cfg),local_files_only=True)
    with init_empty_weights():
        model=Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
        adapter,details=attach_lora(model,cfg,run,meta=True)
    return dict(target_modules=details["target_count"],trainable_parameter_count=details["trainable_parameter_count"],
                actual_structure_inspected=True,weights_loaded=False,gpu_training_verified=False,
                note="LoRA attached to actual meta model modules; not a trained adapter/checkpoint")

def load_student(cfg):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration.from_pretrained(snapshot_path(cfg),
        local_files_only=True,dtype=torch.bfloat16,
        device_map={"":0},attn_implementation="sdpa",low_cpu_mem_usage=True)

def record_phase(run,phase,optimizer_steps,**details):
    import torch
    record=dict(phase=phase,optimizer_steps=optimizer_steps,**details)
    if torch.cuda.is_initialized():
        free,total=torch.cuda.mem_get_info(0)
        record.update(free_bytes=free,total_bytes=total,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(run/"execution_phase.json",record)
    with (run/"logs/phases.jsonl").open("a") as log: log.write(json.dumps(record)+"\n")

def train(cfg,run):
    import torch
    from transformers import set_seed
    manifest=read_manifest(cfg,run);rows=manifest["train"]
    gpu=require_gpu(cfg);set_seed(cfg["seed"])
    # Re-run every sample's mask/length check in this training run before loading weights.
    validate(cfg,run);p=processor_for(cfg)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    record_phase(run,"model_loading",0)
    model=load_student(cfg)
    record_phase(run,"lora_attachment",0)
    model,data=attach_lora(model,cfg,run)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads();model.train()
    params=[param for param in model.parameters() if param.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    rng=random.Random(cfg["seed"]);order=[];epoch=0;exposures=Counter()
    input_tokens=0;supervised_tokens=0;steps=[]
    optimizer.zero_grad(set_to_none=True)
    with (run/"logs/train_steps.jsonl").open("x",buffering=1) as log:
        for step in range(1,cfg["optimizer_steps"]+1):
            lr=cfg["learning_rate"]*min(step/cfg["warmup_steps"],1)
            for group in optimizer.param_groups: group["lr"]=lr
            micro_losses=[];seen=[]
            for micro in range(cfg["gradient_accumulation"]):
                if not order:
                    order=list(range(len(rows)));rng.shuffle(order);epoch+=1
                row=rows[order.pop(0)]
                batch,detail=encode(row,p,cfg)
                record_phase(run,"batch_transfer",step-1,micro_batch=micro+1)
                batch={key:value.to("cuda:0") for key,value in batch.items()}
                record_phase(run,"forward",step-1,micro_batch=micro+1)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    loss=model(**batch).loss
                if not torch.isfinite(loss): raise FloatingPointError("NaN/Inf training loss")
                record_phase(run,"backward",step-1,micro_batch=micro+1)
                (loss/cfg["gradient_accumulation"]).backward()
                micro_losses.append(float(loss.detach()))
                exposures[row["original_id"]]+=1;seen.append(row["original_id"])
                input_tokens+=detail["total_tokens"];supervised_tokens+=detail["supervised_tokens"]
                save(run/"progress.json",dict(optimizer_steps=step-1,
                    completed_micro_batches_in_current_update=micro+1,
                    question_exposures=sum(exposures.values()),unique_questions_exposed=len(exposures),
                    per_question_exposures=dict(exposures),input_tokens=input_tokens,
                    supervised_tokens=supervised_tokens,steps=steps,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                del batch,loss
            record_phase(run,"gradient_clipping",step-1)
            norm=torch.nn.utils.clip_grad_norm_(params,cfg["max_grad_norm"],error_if_nonfinite=True)
            if not any(param.grad is not None for param in params): raise ValueError("No LoRA gradients")
            record_phase(run,"optimizer_step",step-1)
            optimizer.step()
            record_phase(run,"optimizer_update_completed",step)
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
            metric=dict(optimizer_step=step,micro_batches=cfg["gradient_accumulation"],
                mean_micro_loss=sum(micro_losses)/len(micro_losses),micro_losses=micro_losses,
                grad_norm=float(norm),learning_rate=lr,input_tokens_cumulative=input_tokens,
                supervised_tokens_cumulative=supervised_tokens,question_exposures=sum(exposures.values()),
                unique_questions_exposed=len(exposures),question_ids=seen,epoch_started=epoch,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                elapsed_seconds=time.perf_counter()-started)
            steps.append(metric);log.write(json.dumps(metric)+"\n")
            save(run/"progress.json",dict(optimizer_steps=step,question_exposures=sum(exposures.values()),
                unique_questions_exposed=len(exposures),steps=steps,
                peak_allocated_bytes=metric["peak_allocated_bytes"],peak_reserved_bytes=metric["peak_reserved_bytes"],
                input_tokens=input_tokens,supervised_tokens=supervised_tokens))
            print("OPTIMIZER_STEP",json.dumps(metric),flush=True)
    if not any(bool(torch.count_nonzero(param).item()) for name,param in model.named_parameters() if "lora_B" in name):
        raise ValueError("LoRA B remains zero: no verified update")
    record_phase(run,"checkpoint_save",len(steps))
    checkpoint=run/"adapter"
    model.save_pretrained(checkpoint,safe_serialization=True);p.save_pretrained(checkpoint)
    torch.save(dict(optimizer=optimizer.state_dict(),optimizer_steps=len(steps),
                   python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state()),run/"optimizer_state.pt")
    hashes={f.name:sha(f) for f in checkpoint.iterdir() if f.is_file()}
    save(run/"checkpoint_manifest.json",dict(path=str(checkpoint),files_sha256=hashes,
        base_model_id=cfg["model_id"],base_revision=cfg["model_revision"],
        data_manifest_sha256=sha(run/"data_manifest.json"),optimizer_steps=len(steps)))
    save(run/"exposures.json",dict(per_question=dict(exposures),exposure_histogram=dict(Counter(exposures.values())),
         total=sum(exposures.values()),unique=len(exposures),epochs_equivalent=sum(exposures.values())/len(rows)))
    return dict(optimizer_steps=len(steps),adapter_path=str(checkpoint),question_exposures=sum(exposures.values()),
        unique_questions_exposed=len(exposures),epochs_equivalent=sum(exposures.values())/64,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        training_seconds=time.perf_counter()-started,first_loss=steps[0]["mean_micro_loss"],
        last_loss=steps[-1]["mean_micro_loss"],trainable_parameter_count=data["trainable_parameter_count"],
        actual_gpu=gpu,adapter_saved=True,reload_verified=False)

def evaluate(cfg,run):
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import set_seed
    manifest=read_manifest(cfg,run);rows=manifest["check"];require_gpu(cfg);set_seed(cfg["seed"])
    path=cfg.get("adapter_path")
    if not path or not (Path(path)/"adapter_model.safetensors").is_file():
        raise Blocked("No completed P03 adapter checkpoint exists")
    parent=Path(path).parent
    training_metrics=json.loads((parent/"metrics.json").read_text())
    if training_metrics.get("optimizer_steps")!=20 or training_metrics.get("status")!="completed":
        raise Blocked("Adapter is not from a completed20 update P03")
    saved_manifest=json.loads((parent/"checkpoint_manifest.json").read_text())
    if saved_manifest["base_model_id"]!=cfg["model_id"] or saved_manifest["base_revision"]!=cfg["model_revision"]:
        raise Blocked("Adapter base model identity mismatch")
    for name,expected in saved_manifest["files_sha256"].items():
        if sha(Path(path)/name)!=expected: raise Blocked("Adapter artifact checksum mismatch: "+name)
    if sha(run/"data_manifest.json")!=saved_manifest["data_manifest_sha256"]:
        raise Blocked("Training/check data contract changed")
    p=processor_for(cfg);results=[];started=time.perf_counter()
    # Base and adapter are loaded sequentially in a process separate from training.
    for mode in ["base","adapter"]:
        model=load_student(cfg)
        if mode=="adapter": model=PeftModel.from_pretrained(model,path,is_trainable=False)
        model.eval()
        model.config.use_cache=True
        for row in rows:
            with Image.open(row["image"]) as im: image=im.convert("RGB")
            text=p.apply_chat_template(user_messages(row),tokenize=False,add_generation_prompt=True)
            inputs=p(text=[text],images=[image],return_tensors="pt",padding=False,truncation=False)
            n=inputs["input_ids"].shape[1]
            if n+cfg["generation_max_new_tokens"]>cfg["max_sequence_length"]:
                raise ValueError("Generation context exceeds P03 cap")
            if sha(row["image"])!=row["image_sha256"]: raise ValueError("Check image hash mismatch")
            inputs={k:v.to("cuda:0") for k,v in inputs.items()}
            set_seed(cfg["seed"]);stamp=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(**inputs,do_sample=False,num_beams=1,
                    max_new_tokens=cfg["generation_max_new_tokens"],use_cache=True)
            torch.cuda.synchronize()
            generated=output[0,n:];response=p.decode(generated,skip_special_tokens=True)
            answer=response.strip()
            result=dict(mode=mode,original_id=row["original_id"],chart_name=row["chart_name"],
                question=row["question"],gold=row["answer"],response_raw=response,
                extracted_answer=answer,correct=answer.casefold()==row["answer"].strip().casefold(),
                scoring="exact stripped casefold response; no numeric tolerance",
                input_tokens=n,output_tokens=len(generated),
                output_cap_reached=len(generated)>=cfg["generation_max_new_tokens"],
                parse_failure=not bool(answer),elapsed_seconds=time.perf_counter()-stamp,
                image_sha256=row["image_sha256"],generation=dict(do_sample=False,num_beams=1,max_new_tokens=cfg["generation_max_new_tokens"]),
                rendered_prompt=text)
            results.append(result)
            with (run/"responses.jsonl").open("a") as f:f.write(json.dumps(result,ensure_ascii=False)+"\n")
            print("CHECK_GENERATION",mode,row["chart_name"],response,flush=True)
            del inputs,output,generated
        del model;gc.collect();torch.cuda.empty_cache()
    scores={mode:dict(correct=sum(r["correct"] for r in results if r["mode"]==mode),total=16)
            for mode in ["base","adapter"]}
    return dict(reload_verified=True,generation_count=len(results),scores=scores,
                elapsed_seconds=time.perf_counter()-started,
                interpretation="small pipeline check only; no improvement claim")
