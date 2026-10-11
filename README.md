# MMDL experiment code and reports

P03/P04 experiment implementation and sanitized reports for Qwen/Qwen3-VL-4B-Instruct. No teacher or KD is used in these SFT checks. [Report index](INDEX.md).

## Code

- `code/p03/`: recorded ChartQA SFT, assistant-only masks, actual language-module LoRA selection and GPU admission.
- `code/p04/`: E1/E2 preparation, frozen recent-vLLM rescoring, independent LoRA training, adapter-only evaluation and sequential failure-stopping launcher.
- `code/reporting/`: allowlisted report export and non-force publication.
- `configs/` and `scripts/`: actual experiment settings and host launch/status/stop commands.
- [Host runbook](docs/experiments/P04_E12_HOST_RUNBOOK.md).
- `CODE_MANIFEST.json`: SHA256 of each uploaded source file.

## Execution prerequisites

These files are a source snapshot from `/home/user/MMDL`, not a self-contained dataset/model bundle. Existing scripts/configurations retain that host workspace path. Use them in that workspace, or deliberately update paths for another installation. Training uses the existing isolated `.venv-p03`; evaluation setup creates a separate pinned `.venv-p04-vllm`.

Private prerequisites must remain local: source datasets/images, processed QA manifests, pinned model weights, successful P03 adapter for compatibility checks, historical MMMU900 raw responses and evaluation contract, and verified pinned upstream preprocessing/guard sources under `review_snapshots/`. They are intentionally absent from this repository. The launcher refuses incomplete or mismatched contracts. Do not upload those artifacts to satisfy prerequisites.

```bash
cd /home/user/MMDL
bash scripts/setup_p04_eval_env.sh && bash scripts/run_p04_e12.sh
```

The suite performs a separate smoke adapter, then E1 and E2 independently from original weights. It reuses saved-response B0 and never regenerates original-model MMMU answers. E3/PGPS9K is not required by this suite. Existing completed stages are reused; failures stop dependent work.

## Actual status at source publication

P0364-QA/20-update training and reload were completed earlier. E1/E2 data and CPU masks/contracts were validated. This code publication does not mean E1/E2 GPU training/evaluation completed. Inspect local `runs/<run_id>/metrics.json` and the report index for actual execution status. Mixed563/900 and uniformly rescored491/900 are separate scoring results on historical responses, not a training improvement comparison.

Datasets, images, weights, full responses, large logs and credentials are excluded. Original dataset/upstream licenses and attribution still apply; no new blanket license is assigned here.
