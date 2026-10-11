# E1/E2 host execution, 2026-10-11

This supersedes prerequisite status in P04_LATEST_VLLM_STATUS_20261011.md without changing its historical record.

## Completed prerequisites

- Host CUDA tensor test passed: runs/P04_20261011T034748Z_host_gpu_probe.
- Recent MMMU export completed: runs/P04_20261011T035153Z_acquire_recent_mmmu; 900 questions, 982 decoded images. Manifest SHA256 eed2e5d5327debaa7602eb67dbcd12cf7e68358c62f464a170bd196bcf420697 exactly matches the historical vLLM export. Image numbers/order, annotations and individual image hashes were checked.
- Readiness passed: runs/P04_20261011T035901Z_ready_B0. Data: datasets/processed/P04_E12_v1. E1=ChartQA1000; E2=500 of E1 plus TabMWP500. Independent checks=100 per source. No exact question/RGB match or dHash-distance<=2 image candidates against recovered MMMU images. Semantic duplication is not exhaustively ruled out.
- B0 is CPU rescoring of existing recent vLLM responses, runs/P04_20261011T033804Z_latest_vllm_B0_v2:491/900. Historical mixed563/900 remains separate. Original-model inference is prohibited.
- CPU parser21 cases and pipeline9 tests passed: runs/P04_20261011T040555Z_selftest_B0. Earlier processor/mask validation covered1700 unique QA; no truncation/padding. GPU adapter training/inference with the new suite is not yet verified.
- PGPS9K download failure was reported by the user. Only preprocessing-source files are present locally; original ZIP/images have not been acquired. Exact host download error has not been supplied. E3 remains unprepared; it is not an E1/E2 prerequisite. No automatic PGPS9K retry is part of these commands.

## Commands in the basic host terminal

Do not start these while another GPU training/inference task is running. Admission checks are read-only and preserve other processes. Do not rerun completed image downloads or regenerate B0.

```bash
cd /home/user/MMDL
bash scripts/setup_p04_eval_env.sh
```

The setup creates a separate owned .venv-p04-vllm with versions from the actual recent run, then verifies vLLM's real Qwen3VL mapping against504 P03 adapter tensor names. It does not replace .venv-p03 or change drivers. A package/version/network/disk failure is recorded and stops execution; do not continue after setup failure. Available disk was23GiB at preparation; installer requires12GiB reserve before creating this environment. Existing caches/models/data are preserved.

After successful setup:

```bash
mkdir -p logs
nohup bash scripts/run_p04_e12.sh > logs/P04_E12_host.log 2>&1 &
```

Order: data readiness → pinned eval environment check → CPU tests → separate8-QA/2-update smoke training → separate-process reload → adapter-only16-question smoke inference → E1 original initialization/1000 QA/250 updates → reload → MMMU900+ChartQA100+TabMWP100 → E2 independent original initialization/same250 updates → reload → same1100 questions → paired comparison. Smoke weights are never continued. One pass means1000 question exposures per main run. Warmup13 updates, then constant1e-4; identical BF16 LoRA/image/optimizer conditions for E1/E2.

Historical900-question inference took19097.93seconds (5h18m). Each adapter evaluates1100 questions, so reserve several hours per adapter; exact duration and training estimates remain unmeasured until smoke/main logs exist. Setup network/package time is additional. No automatic condition adjustment or E3 launch occurs.

## Status, stop and explicit resume

```bash
bash scripts/p04_e12_status.sh
tail -f logs/P04_E12_host.log
bash scripts/stop_p04_e12.sh
```

The stop script verifies the owned suite PID and sends SIGTERM only to that suite and its newly created stage process group. It does not terminate unrelated sessions or GPU processes. If termination is still pending, inspect the recorded group before restarting.

After reviewing a failure and fixing its cause without changing the frozen comparison contract:

```bash
bash scripts/run_p04_e12.sh --retry-failed
```

Completed stages are reused; failed evaluation responses resume only when every contract hash matches. An incomplete training stage starts a new run independently from the original model; partial updates are preserved but not advertised as resumable optimizer state. Changes to comparison code/config stop automatic reuse and require deliberate new-suite handling. Publication errors do not erase or fail a successful learning/evaluation stage. GPU/computation failures stop every dependent stage.

## Frozen evaluation and limitations

configs/P04/E12_latest_vllm.json points to B0's evaluation_protocol.json. The original saved messages are reused byte-for-byte except verified local image paths. Evaluation uses vLLM0.11.2,8192 output tokens,49152 context, pixels1003520/4014080, temperature0.7/top_p0.8/top_k20/presence_penalty1.5/repetition_penalty1.0, seed3407, and unchanged original repetition guard. P03's64-token generation cap is not used. Training pixels remain200704/1003520, max sequence4096, dynamic unpadded encoding. User/image tokens are excluded from loss.

Adapters require dynamic LoRA enablement and BF16 LoRA kernels; this necessary backend deviation is recorded. Expanded CPU/vLLM prompt tokens must match historical counts before a score is accepted. No GT is passed into message construction or extraction; GT comparison happens afterward. Cap reach, missing final answer, parse failure and repetition abort are distinct; cap reach alone is not incorrect.

Every stage creates experiment.md, actual config/environment/data hashes, commands, metrics and logs. Raw responses remain private. Sanitized settings/aggregates and MMMU ID-level decisions export and publish sequentially to Hyeonjin1/mmdl through the existing publisher. Authentication/network failures are recorded separately; a successful export is not a successful publication.

No reviewed work-type label file is available. Task-specific and non-target changes cannot be claimed by substituting subject names. All paired IDs/subject results are retained for later reviewed label joins. No newly inferred original-model dataset-check scores are available, so adapter heldout scores have no B0 dataset-check improvement baseline. This is one-seed exploration using MMMU validation as a development set, not official-score or replicated-improvement evidence.
