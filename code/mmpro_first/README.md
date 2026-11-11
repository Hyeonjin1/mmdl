# Qwen3-VL weighted SFT → hard negatives → DPO/ranking

The first run is a bounded pipeline smoke test, not a benchmark improvement claim.
All raw teacher traces, prompts containing dataset text, full model generations,
mask dumps, preference caches and adapters stay in ignored local directories.

Use `.venv-mmpro-first/bin/python`. This overlay uses the existing torch 2.7.0+cu118,
Transformers 4.57.1 and PEFT 0.17.1 installation without changing it; PyArrow 18.1.0
was installed in the overlay. Model, processor and tokenizer use revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. The RAM model cache is volatile.

## New run commands

Run from `/home/user/mmdl-work`. These commands create separate directories and
refuse to replace existing preparation directories. Expanded mode is a bounded
pool of up to 128 items per source, not the complete 123K/874K corpora. It uses
128 SFT updates (384 exposures; balanced source schedule) and at most 32 mined
questions/16 preference updates. Missing or rejected source examples are reported;
a short source pool can repeat. No GRPO is implemented.

```bash
PY=/home/user/mmdl-work/.venv-mmpro-first/bin/python
$PY code/mmpro_first/launch.py prepare --run-id MMProFirst_expanded_01 --profile expanded
# Run in tmux to preserve execution across terminal disconnects.
tmux new-session -d -s mmpro-expanded \
  'cd /home/user/mmdl-work && .venv-mmpro-first/bin/python -u code/mmpro_first/launch.py train --run-id MMProFirst_expanded_01 > .restore/MMProFirst_expanded_01/launcher.log 2>&1'
tail -f .restore/MMProFirst_expanded_01/launcher.log
```

`prepare` checks free space before downloading pinned Parquet shards and streams
ArxivQA archive images. The expanded profile caches both complete MMMU-Pro
settings in advance for overlap checks. `train` evaluates the fixed ArxivQA dev
set for all five models, locks selection there, and only then runs the fixed Pro
input/scoring smoke test (at most 32 items per setting). It does not evaluate the
outer ArxivQA test partition or the full Pro benchmark automatically.

After the configuration and model are locked, the following **separate, expensive**
command materializes outer ArxivQA test images and evaluates that split plus both
complete MMMU-Pro settings. It must not be used for checkpoint selection:

```bash
$PY code/mmpro_first/launch.py final --run-id MMProFirst_expanded_01
```

This command is supplied for later execution; the first smoke run does not execute
it. Full preparation/training/final evaluation has not been validated at full scale.
Run ID directories preserve completed stages and failed attempts. Inspect `.restore/<ID>/*.log`
and `runs/<ID>/*/complete.json`; stage receipts include exact code/config/data hashes.
A source or mining validation failure stops the chain instead of inventing data.

## Loss, token boundaries and comparison

Microbatch 1 with accumulation implements batch means of **samplewise** region
means, not token-count weighted mixing. The default is
`(mean(reason) + 3*mean(answer) + mean(format))/5`.
Image/prompt/padding tokens receive no supervision; close tags and EOS are format.
No packing or truncation is allowed. Whitespace after `<answer>` avoids a real
Qwen tokenizer merge of `>B`; label scores sum exactly the actual label tokens.

`run.py mixed_w1` implements the same initialization/data/order/updates with only
the answer coefficient changed to 1; its denominator remains 5 to isolate the
coefficient. This optional ablation is not automatically included in the first
five-model comparison. Use a new run ID for alternative settings. If a convex
normalized `(r+a+f)/3` objective is desired, register it as a separate condition;
it also changes the global gradient scale.

Mining generates reasoning without gold, then fixes a shared reasoning/answer
prefix. All actual option labels are scored. Top two non-gold labels form both
DPO pairs and the same ranking candidate set. DPO averages negative pairs **within
question** before averaging questions. Its frozen reference scores the entire
completion suffix including closing tag/EOS; ranking uses label log-prob sums
only. Cache contracts hash actual input IDs, completion masks, pixels, image grid,
source IDs, adapter, processor and prefix. Reference scores are recomputed before
training. Each preference method starts from the same mixed-SFT checkpoint.

Strict free-generation two-tag accuracy and independently generated-reasoning
label likelihood accuracy are separate metrics and incur separate generation
costs. Failed formats and length caps remain in the denominator. Official MMMU-Pro
parser accuracy is a separate smoke metric, with its random fallback retained.
The user-requested two-tag prompt differs from the official prompt, so this is
not an exact reproduction of the official leaderboard protocol. No answer or
explanation is supplied to the input builder.

## Data caveats

ArxivQA paper groups use seed 42 for outer 80/20, with 5% of outer-train groups
reserved as dev. The smoke image subset follows deterministic archive order,
so it is not a random representative sample. The smoke Pro cache is the first
32 Parquet rows of each setting, spanning several subjects but not a balanced sample. Exact text, source/provenance and
available decoded-image SHA256 overlap checks are recorded with their scope.
There is no claim that unavailable image hashes or perceptual duplicates were
checked. OpenMMReasoner's chosen config provides an assistant answer rather than
a separate gold field; parsing and option mapping are verified, not independently
certified correctness. Teacher final-answer mismatches and overly long targets
are rejected instead of repaired/truncated.

`checks.py` supplies independent loss/label/parser/DPO/ranking checks. GPU
`preflight` performs real image forward/backward with an adapter discarded before
training. FlashAttention2 is attempted once per run; SDPA fallback is recorded.

To reproduce the initial bounded profile under a new ID:

```bash
$PY code/mmpro_first/launch.py prepare --run-id MMProFirst_smoke_replay --profile smoke
$PY code/mmpro_first/launch.py train --run-id MMProFirst_smoke_replay
```

On this host the separate environment inherits the old training environment's
site-packages read-only through a `.pth` file. For a clean environment, install
`torch==2.7.0` from `https://download.pytorch.org/whl/cu118`, then the pinned
`requirements.txt`. The clean installation recipe itself has not been executed;
the overlay environment listed in the report is the one actually validated.
