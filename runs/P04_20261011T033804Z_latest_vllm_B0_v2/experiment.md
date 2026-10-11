# P04_20261011T033804Z_latest_vllm_B0_v2

Purpose: P04 recorded preparation and latest-vLLM evaluation audit. No teacher/KD.

Stage: latest_vllm_B0; status: completed.

Uniform saved-response B0: 491/900 (54.56%). Prior mixed judging:563/900(62.56%); rescoring delta -8.00 percentage points. Extraction failures:130; length cap:60; repetition abort:53. This difference is scoring-policy change on identical responses, not training improvement. Initial493/900 responses were not used.

Actual settings and summary metrics: [config_resolved.json](config_resolved.json), [metrics.json](metrics.json).

Limitations: MMMU validation is a development comparison set. Exact answer and unit-preserving scoring can miss semantic equivalents. No reasoning verification, official score or training improvement claim. Work-type labels are unavailable; subjects are not task types. P03's64-token check is separate from latest-vLLM8192-token evaluation. GPU and matching original images must be verified before dependent E1/E2 execution. E3/PGPS9K availability does not gate E1/E2. Failed prerequisites stop dependent runs.

Public scope: settings, aggregates, this report and optional ID-level judgments only. No raw responses, questions, gold answers, images, original data, checkpoints, logs or credentials. Private source experiment SHA256: ba5d189994ae44e723b710fbcb7d0d29183960e4f38ae702522d07e0da0db1c7.
