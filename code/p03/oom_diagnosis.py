"""Evidence from the preserved failure, plus current read-only GPU snapshot."""
import json
from common import ROOT, read_manifest, save, sha
from gpu_admission import snapshot

def diagnose(cfg,run):
    read_manifest(cfg,run)
    failed=ROOT/"runs/P03_20261008T073430Z_train_bf16"
    names=["experiment.md","config_resolved.json","metrics.json","environment.json","logs/error.json","logs/stage.log"]
    hashes={name:sha(failed/name) for name in names}
    error=json.loads((failed/"logs/error.json").read_text())
    metrics=json.loads((failed/"metrics.json").read_text())
    trace=error["traceback"]
    assert "model=load_student(cfg)" in trace and "_load_state_dict_into_meta_model" in trace
    assert metrics["optimizer_steps"]==0
    save(run/"logs/current_gpu_snapshot.json",snapshot())
    report=dict(failed_run=str(failed),failed_files_sha256=hashes,
        confirmed_failure_phase="model_loading",failed_optimizer_steps=0,
        evidence="from_pretrained -> _load_pretrained_model -> load_shard_file -> _load_state_dict_into_meta_model, before attach_lora/forward/backward",
        failure_message=error["message"],
        other_process_at_failure=dict(pid=1202089,memory_gib_from_error=21.40,memory_mib_from_nvidia_smi=21910),
        own_process_memory_gib_from_error=2.08,free_memory_mib_at_failure=3.62,
        conclusion="Other process occupancy exhausted GPU during weight loading. Idle24GB BF16 training suitability remains untested.",
        historical_gpu_count=1,current_gpu_state="See current_gpu_snapshot.json; historical PID is not assumed still active",
        original_peak_vram_bytes=None,original_peak_vram_limitation="No peak telemetry was saved before model loading failed",
        qlora_decision="Not triggered: failure does not establish BF16 incompatibility on idle24GB. No image/data/precision modification.")
    save(run/"oom_diagnosis.json",report)
    comparisons=[]
    for candidate in sorted((ROOT/"runs").glob("P03_*_validate")):
        c=json.loads((candidate/"config_resolved.json").read_text())
        m=json.loads((candidate/"metrics.json").read_text())
        if c["max_sequence_length"]!=2048 or m["status"]!="completed": continue
        base_cfg=json.loads((failed/"config_resolved.json").read_text())
        ignored={"stage","config_source","purpose","max_sequence_length"}
        assert {k:v for k,v in c.items() if k not in ignored}=={k:v for k,v in base_cfg.items() if k not in ignored}
        base=json.loads((failed/"validation_checks.json").read_text())
        new=json.loads((candidate/"validation_checks.json").read_text())
        assert base==new
        assert json.loads((failed/"mask_samples.json").read_text())==json.loads((candidate/"mask_samples.json").read_text())
        comparisons.append(dict(base_run=str(failed),comparison_run=str(candidate),rows=80,
            sequence_limits=[4096,2048],max_observed_tokens=m["max_total_tokens"],
            all_lengths_grids_and_supervised_counts_equal=True,first2_token_ids_and_masks_equal=True,
            padding=False,truncation=False,target_truncated=False,
            memory_effect="No tensor-size reduction: this limit only rejects overlong samples; no measured CUDA memory comparison"))
    save(run/"sequence_comparison.json",comparisons)
    return dict(diagnosis=report,sequence_comparisons=comparisons,optimizer_steps=0)
