# P03_20261008T081920Z_completion_audit

Purpose: P03 ChartQA gold-answer SFT pipeline validation. This is not KD or a performance-improvement claim.

Stage: completion_audit; terminal status: completed.

Actual settings: see [config_resolved.json](config_resolved.json).
Results: see [metrics.json](metrics.json).

Data: official ChartQA train_human; fixed seed3407;64 distinct training charts and16 different check charts; one original QA per chart. Source revision044eabfc306abfe9340c5741f0093aefc5973d06. Evaluation datasets and corrected evaluation solutions are excluded from training.

Loss covers assistant answers and necessary end tokens only. Language attention/MLP LoRA; vision encoder and merger frozen. For completed20-update training,80 question exposures comprise48 questions seen once and16 seen twice, equivalent to1.25 epochs.

Limitations: base/adapter16-question generation is a pipeline check. Full-response strict strip/casefold matching and64-token generation cap affect the scores. In the completed P03 generation check, base14/16 responses reached the cap; adapter0/16. Cap reached is not the same as missing a final answer. No MMMU evaluation or improvement claim. Prior563/900 combines different judgment rules and is not a matched baseline.

Next step: curate separate training data and validate teacher quality before a separately authorized KD run. Teacher calls and KD training were not started.

Public scope: summary settings and metrics only. Data, images, checkpoints, responses, commands, credentials and large logs are excluded. Private source experiment SHA256: b4df77990421e7772b7fb3c76ec7063456418a6587ebf38c39b205c554971df6.
