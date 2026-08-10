# DominoSearch Agent Guide

## Project objective

The active objective of this repository is to optimize DominoSearch
**layer-wise mixed N:M pruning** so a selected model can be evaluated and later
deployed on resource-constrained hardware, currently targeting Jetson Nano.

Mixed N:M pruning is the only active pruning direction. Uniform N:M, structured
channel/filter pruning, and unstructured magnitude pruning are frozen historical
baselines: preserve their code, branches, artifacts, and reports, but do not add
new implementation or experiments for them unless the user explicitly reopens a
direction. Do not combine their masks or checkpoints with mixed N:M results.

Do not add quantization, distillation, architecture replacement, or another
model-compression method unless the user explicitly requests a separate combined
experiment. Hardware profiling, packed N:M representation, runtime integration,
and kernel work are allowed when they are needed to measure or deploy the same
mixed N:M model, but their effects must be reported separately from pruning-only
accuracy and theoretical-complexity results.

## Read before changing code

Before implementing an optimization, read:

1. `docs/UNDERSTANDING_DOMINOSEARCH.md`
2. `docs/PRUNING_OPTIMIZATION_DIRECTIONS.md`
3. `benchmark/README.md`
4. The relevant search, sparse-operation, model, and training code.

The paper is available at `assets/DominoSearch.pdf`. Use it when a change depends
on the original algorithm or experimental protocol.

## Branch responsibilities

Only one experiment branch is active:

- `master`: shared, stable code, documentation, and benchmark infrastructure.
- `pruning-domino-mixed-nm`: all active DominoSearch layer-wise mixed N:M,
  conditioned sensitivity, measured hardware cost, search, fine-tuning, and
  deployment-evaluation work.

The branches `pruning-uniform-nm`, `pruning-structured-channel`, and
`pruning-unstructured-magnitude` are frozen historical baselines. Do not modify,
run new experiments on, or synchronize them unless the user explicitly reopens
that direction. Their existing results may be used as read-only comparison data.

Shared fixes or benchmark improvements must be committed to `master` first and
then merged into `pruning-domino-mixed-nm` before experiment-specific work. Do
not synchronize frozen branches by default.

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
| `pruning-domino-mixed-nm` | Per-layer mixed N:M, sensitivity, measured cost, search, fine-tuning, and deployment evaluation |
| Any frozen pruning branch | Read-only inspection only unless the user explicitly reopens it |

Before continuing, the agent must be able to answer all of these questions with
`yes`:

- Am I on `pruning-domino-mixed-nm` for experiment work, or `master` for a shared change?
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
`master` first. Synchronize only `pruning-domino-mixed-nm` before continuing. Do
not copy a branch-specific version of shared benchmark code, and do not update
frozen branches by default.

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

- only mixed N:M or shared infrastructure is implemented;
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

Every mixed N:M stage and every retained historical baseline must use the same
dense baseline and conditions when results are compared:

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
- Keep commits focused on mixed N:M or one shared infrastructure change.
- Include the branch name and experiment purpose in result notes or reports.
- Leave unrelated user changes untouched.
