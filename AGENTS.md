# DominoSearch Agent Guide

## Project objective

The primary objective of this repository is to research and compare methods for
optimizing neural-network models through **pruning** so they can run on
resource-constrained hardware such as CPUs, embedded boards, and FPGAs.

Pruning is the main scope. Do not add quantization, distillation, architecture
replacement, or hardware-specific deployment as part of a pruning experiment
unless the user explicitly requests it. Those techniques may be proposed as
future work, but their effects must not be mixed into pruning-only benchmark
results.

## Read before changing code

Before implementing an optimization, read:

1. `docs/UNDERSTANDING_DOMINOSEARCH.md`
2. `docs/PRUNING_OPTIMIZATION_DIRECTIONS.md`
3. `benchmark/README.md`
4. The relevant search, sparse-operation, model, and training code.

The paper is available at `assets/DominoSearch.pdf`. Use it when a change depends
on the original algorithm or experimental protocol.

## Branch responsibilities

Keep each experiment isolated in its assigned branch:

- `master`: shared, stable code, documentation, and benchmark infrastructure.
- `pruning-uniform-nm`: one uniform N:M configuration across sparse layers.
- `pruning-domino-mixed-nm`: DominoSearch mixed N:M selected per layer.
- `pruning-structured-channel`: structured channel or filter pruning.
- `pruning-unstructured-magnitude`: local/global magnitude pruning baseline.
- `pruning-unstructured-gradual`: gradual local/global magnitude pruning with a
  monotonic mask schedule during fine-tuning.

Do not implement one branch's pruning method in another branch. Shared fixes or
benchmark improvements should be committed to `master` first and then merged or
fast-forwarded into all experiment branches before experiment-specific work.

## Mandatory agent self-check

Every agent must run this self-check before editing files. It is not optional.
The agent must state the result in its work update so the user can verify that it
is operating on the correct branch and pruning direction.

### 1. Pre-flight check

Run:

```bash
git branch --show-current
git status --short
git log -1 --oneline
```

Then verify the current branch against this table:

| Current branch | Work that is allowed |
| --- | --- |
| `master` | Shared benchmark, compatibility fixes, documentation, and infrastructure only |
| `pruning-uniform-nm` | Uniform N:M pruning only |
| `pruning-domino-mixed-nm` | Per-layer mixed N:M and DominoSearch cost/search improvements only |
| `pruning-structured-channel` | Structured channel/filter pruning only |
| `pruning-unstructured-magnitude` | Local/global magnitude pruning only |
| `pruning-unstructured-gradual` | Gradual local/global magnitude pruning only |

Before continuing, the agent must be able to answer all of these questions with
`yes`:

- Am I on the branch assigned to this pruning method?
- Is the requested work inside that branch's allowed scope?
- Have I read the relevant documentation and implementation?
- Is the worktree clean, or have I identified and preserved every existing user
  change that overlaps my task?
- Do I know which dense checkpoint and benchmark form the baseline?
- Can the new method be measured independently from other optimization methods?

If any answer is `no`, stop implementation and resolve the mismatch first. Do not
silently switch methods, discard changes, or broaden the experiment.

### 2. Scope check before each material edit

Before changing a file, confirm:

- why the file must change for the current branch's method;
- whether the change belongs on `master` as shared infrastructure instead;
- whether it changes original DominoSearch behavior;
- whether a new flag/module can preserve the original default behavior;
- which test or benchmark will detect an incorrect implementation.

If a change is useful to more than one pruning branch, implement and commit it on
`master` first. Synchronize the experiment branches before continuing. Do not
copy slightly different versions of shared benchmark code into each branch.

### 3. Experiment validity check

Before running or accepting a pruning experiment, confirm:

- the dense and pruned runs use the same architecture, dataset split,
  preprocessing, input size, device, and benchmark settings;
- the checkpoint loads without missing or unexpected keys;
- the pruning scheme/mask covers exactly the intended layers;
- measured sparsity matches the requested sparsity;
- both pre-fine-tuning and post-fine-tuning accuracy are recorded;
- the output JSON records branch, commit, seed, checkpoint, and scheme/mask;
- theoretical MAC/parameter reductions are not presented as hardware speedup.

An experiment that fails one of these checks is a debug run, not valid evidence.
Label it accordingly and do not include it in the final comparison table.

### 4. Pre-commit check

Run at minimum:

```bash
git branch --show-current
git status --short
git diff --check
git diff --stat
```

Review the full diff and verify:

- only the current branch's pruning direction is implemented;
- no checkpoint, dataset, cache, log, or generated result is staged;
- original dense behavior still works;
- documentation and CLI help match the implementation;
- relevant syntax checks, smoke tests, and benchmark checks pass.

Do not commit if the branch name and implemented method do not match.

### 5. Handoff self-audit

Before reporting completion, the agent must explicitly report:

- current branch and commit;
- pruning method implemented;
- files changed;
- checks and benchmarks that passed;
- dense baseline and comparable pruned result, if an experiment was run;
- tests that could not run and the reason;
- whether results prove theoretical reduction, host runtime improvement, or
  target-hardware improvement.

The agent must not say "optimized", "faster", or "best" unless the corresponding
baseline evidence exists. If only code was implemented, say that the method is
ready for evaluation, not that it has already improved the model.

## Preserve the original project

- Preserve the original DominoSearch algorithm and default behavior unless the
  task explicitly asks to change them.
- Prefer adding isolated modules, flags, configs, or scripts over rewriting the
  original research implementation.
- Clearly document intentional behavioral or numerical changes.
- Never overwrite checkpoints, schemes, datasets, logs, or benchmark results.
- Do not commit generated datasets, model checkpoints, cache files, or large
  benchmark artifacts.

## Experimental fairness

Every pruning direction must start from the same dense baseline and must use the
same conditions when results are compared:

- model architecture and dense checkpoint;
- training and validation dataset/split;
- preprocessing and input size;
- random seeds;
- fine-tuning budget and evaluation method;
- benchmark device and software versions;
- performance batch size, warm-up count, and measured iteration count.

For each experiment, preserve both results:

1. pruned before fine-tuning;
2. pruned after fine-tuning.

At minimum report:

- Top-1 and Top-5 accuracy;
- dense and effective parameters;
- dense and effective MACs;
- median and P95 latency;
- throughput;
- peak device memory when available;
- pruning scheme/mask and checkpoint identity.

Use `benchmark/benchmark_model.py` to produce JSON results and
`benchmark/compare_results.py` to create comparable tables.

## Interpreting sparse performance

The current PyTorch sparse layers generate masks but call dense PyTorch
operators. Consequently:

- effective parameter and MAC reductions are theoretical N:M metrics;
- PyTorch checkpoint size does not automatically shrink;
- lower effective MACs do not prove lower runtime latency;
- Colab GPU or CPU latency does not prove FPGA speedup.

Claims about FPGA or board performance require measurements on the target
hardware. Report latency, throughput, power/energy, BRAM, DSP, LUT, bandwidth,
and binary/model size when those measurements become available.

## Implementation rules

- Validate all pruning ratios and fail clearly when `N:M` is invalid.
- Validate that a pruning scheme matches every expected sparse layer. Do not
  silently ignore missing or unknown layers.
- Validate checkpoint keys. Do not benchmark a partially loaded model unless the
  user explicitly authorizes it and the report clearly identifies the mismatch.
- Keep dense baseline behavior available for every new pruning implementation.
- Make experiment outputs reproducible by recording arguments, seeds,
  environment details, checkpoint paths, and scheme/mask information.
- Keep training, evaluation, and benchmark code device-aware; do not assume an
  NVIDIA GPU is present.
- Avoid introducing dependencies unless they are necessary and documented in
  `requirements.txt`.

## Verification before handoff

Run checks proportional to the change. At minimum:

1. run Python syntax/compile checks for modified Python files;
2. run `git diff --check`;
3. run a small synthetic smoke test when PyTorch is available;
4. run a validation subset before any full dataset experiment;
5. compare the resulting JSON against the dense baseline;
6. state clearly which tests could not run and why.

Do not report an optimization as successful without baseline evidence. Prefer a
Pareto comparison across accuracy, effective complexity, latency, and memory over
selecting a model from sparsity alone.

## Documentation and commits

- Update documentation when adding a CLI flag, result field, pruning rule, or
  experiment protocol.
- Keep commits focused on one pruning direction or one shared infrastructure
  change.
- Include the branch name and experiment purpose in result notes or reports.
- Leave unrelated user changes untouched.
