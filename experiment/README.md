# XGrammar compiled-workload benchmark

`run_xgrammar_benchmark.py` runs XGrammar against one current 500-instance,
schema-v4 compiled workload. It contains no constraint-family dispatch:
every job is driven exclusively by its authenticated `compiled_constraint`
(token partition, NFA, length bounds, and exact prompt token IDs).

The CommonGen and CoAuthor workloads are run separately. From this repository:

```bash
PYTHON=/project/aip-ksmeel/sunjia72/miniconda3/envs/nfa/bin/python
NFA=/project/aip-ksmeel/sunjia72/constraint_decoding/nfa_fpras

$PYTHON experiment/run_xgrammar_benchmark.py \
  --workload "$NFA/experiment/results_nfa_updated_hmm_full500_20260728/qwen_common_gen/workload.json" \
  --local_files_only --gpus 0,1,2,3 --timeout_s 256

$PYTHON experiment/run_xgrammar_benchmark.py \
  --workload "$NFA/experiment/results_nfa_updated_hmm_full500_20260728/qwen_coauthor/workload.json" \
  --local_files_only --gpus 0,1,2,3 --timeout_s 256
```

Replace the two `qwen_*` workload directories with `gemma_*` (and pass the
matching Gemma model path) for the Gemma runs.

The output run contains a byte-identical `workload.json`, `manifest.json`,
`plan.csv`, `results.csv`, `summary.json`, and isolated per-job attempt logs.
Successful and failed jobs are checkpointed atomically.

Resume without rerunning terminal jobs:

```bash
$PYTHON experiment/run_xgrammar_benchmark.py \
  --workload /path/to/workload.json \
  --local_files_only --gpus 0,1,2,3 --timeout_s 256 \
  --resume --run_dir /path/to/existing/run
```

Add `--retry_failed` to rerun failed jobs. `--indices 0-9` and `--max_jobs N`
support small executions.

`--timeout_s 256` is a shorthand that gives each job two independent
256-second budgets: one through the authenticated precompute checkpoint, and
one after that checkpoint for mask auditing, model loading, and generation.
Use `--precompute_timeout_s` and `--generation_timeout_s` to set those budgets
separately. A post-precompute timeout retains the measured precomputation
metrics in `results.csv`.

## Non-model validation

Authenticate and summarize a workload:

```bash
$PYTHON experiment/run_xgrammar_benchmark.py \
  --workload /path/to/workload.json --dry_run
```

Compile and audit the maximum-unrolled endpoint of each of the ten families
without loading model weights:

```bash
$PYTHON experiment/run_xgrammar_benchmark.py \
  --workload /path/to/workload.json \
  --pilot_grammar --local_files_only
```

## Semantics and timing

Each NFA symbol class is mapped to one synthetic Unicode character in a custom
XGrammar tokenizer. Every real, nonspecial model token remains available in
exactly one class; non-EOS special tokens and logits-only padding IDs are
masked, and EOS is the sole stop token. Each successful worker independently
checks the generated token IDs against the compiled NFA and audits XGrammar's
initial full-vocabulary mask.

`xgrammar_precompute_runtime_s` (also exposed as `setup_runtime_s`) measures
compiled-NFA/token-partition conversion to bounded GBNF plus uncached XGrammar
compilation. It excludes model loading, mask auditing, and generation.
