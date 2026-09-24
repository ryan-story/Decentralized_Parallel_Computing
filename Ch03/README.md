# Chapter 3: Synchronizing Without Global Barriers

The code for Chapter 3 of *Decentralized Parallel Programming*. It computes a mean
across sixteen workers that hold unequal batches of data, first as an ordinary
distributed reduction that needs every worker, then as commit-and-merge, which
commits once enough workers have contributed and says exactly which ones did.

Section 3.2.1 of the book refers to this folder,
`Decentralized_Parallel_Programming/Ch03/`, as the chapter's companion code.

## The files, in the order the chapter builds them

| file | chapter section | listings | what it is |
|---|---|---|---|
| `meanreductionkernel.cu` | 3.1.1, 3.1.2 | 3.2, 3.3 | `MeanReductionKernel`, which finalizes a mean inside one block, and `PartialMeanKernel`, which returns `(sum, count)` and never divides. The only device code in the chapter. |
| `partial_launcher.py` | 3.3.1 | 3.4 | `partial_launcher`: one worker's whole local job. Cuts a shard into power-of-two blocks of at most 1,024 records, reduces each, and merges the block partials into one contribution. |
| `distributed_mean_coordinator.py` | 3.3.1 | 3.5 | The all-participant coordinator. Shards the input, runs every worker at once, waits for all of them, merges, divides once. |
| `all_reduce_coordinator.py` | 3.3.2 | 3.6 | The same second reduction as an NCCL All-Reduce, so every GPU ends up holding the global sum and count. |
| `commit_merge.py` | 3.4.1 to 3.4.6 | 3.7 to 3.13 | The commit-and-merge protocol: `EpochConfig`, `Contribution` and the status records, `CommitMergeEpoch` with its admission predicate, close rules and `freeze`, and `canonical_merge`. |
| `commit_merge_coordinator.py` | 3.4.7 | 3.14, 3.15 | `run_worker` packages a worker's reduction as a `Contribution`; `run_commit_merge` runs a whole epoch on the wall clock. |
| `population.py` | 3.2 | | The working example's sixteen workers and their data, generated here, plus seeded completion times. |
| `epoch_experiments.py` | 3.2 to 3.5 | | The experiment harness: drives `CommitMergeEpoch` through thousands of seeded epochs and returns each result the chapter reports. |
| `experiment_harness.ipynb` | | | Runs the harness and draws the chapter's results: figures 3.3 to 3.8, and tables 3.4, 3.5, 3.6, 3.8, 3.10 and 3.11. Nothing else. |

Listing 3.1 is pseudocode, and figures 3.1, 3.2 and 3.9 are diagrams, so none of
them has a file. Figure 3.6 is drawn by replaying its seven arrivals through
`CommitMergeEpoch`, so its colours are the statuses the protocol records.

## Getting started

```bash
pip install -r requirements.txt
```

A GPU is optional. With an NVIDIA device and CuPy installed,
`meanreductionkernel.cu` is compiled at run time and every worker's reduction runs
there; without one, the same halving tree runs on the CPU through NumPy. Every
entry point prints which path it took.

**Reproduce the book's numbers:**

```bash
jupyter lab experiment_harness.ipynb
```

It runs in well under a minute on either path and reproduces every table value
the chapter quotes. Each worker's `(sum, count)` is computed by the chapter's
kernel through `partial_launcher`, and the protocol is `commit_merge.py`
unchanged; only the arrival times and dropout are seeded draws.

**Run the listings directly:**

```bash
python distributed_mean_coordinator.py     # listing 3.5
python commit_merge_coordinator.py         # listings 3.14 and 3.15, on the wall clock
```

## Running on fewer GPUs than the book assumes

Listing 3.5 is written for four GPUs, `device_ids = [0, 1, 2, 3]`, and listing
3.15 for one GPU per worker. Device IDs may repeat: on a one-GPU machine,
`device_ids = [0, 0, 0, 0]` gives four workers that take turns on the same device.
The arithmetic is identical; only the concurrency differs.

Listing 3.6 uses NCCL, which runs on Linux with NVIDIA GPUs and allows one rank per
GPU, so its `device_ids` must be distinct (`[0]` on a one-GPU machine).
`all_reduce_coordinator.nccl_available()` reports whether it can run. Where NCCL is
missing, including Windows and macOS, `all_reduce_mean` reduces each shard with
`partial_launcher` and adds the pairs on the host instead, returning one mean per
rank as the collective would. The result is the same; there is no collective.

## Where the code differs from the book's listings

The function and class bodies are the listings'. The differences are only what it
takes to turn listings into importable modules:

- `meanreductionkernel.cu` declares both kernels `extern "C"`, so CuPy can look them
  up by name.
- `partial_launcher.py` falls back to NumPy when CuPy or a GPU is missing, and adds
  two small helpers: `gpu_available()`, and `mean_kernel()`, which launches listing
  3.2 so the notebook can show it. On the CPU path each block's tree matches the
  kernel bit for bit; adding the block partials can differ in the last bits,
  because `cp.sum` picks its own order.
- `distributed_mean_coordinator.py` wraps listing 3.5's program in
  `distributed_mean(v, device_ids)`. Running the file still runs the program.
- `commit_merge.py` writes listings 3.10 to 3.12 inside the one `CommitMergeEpoch`
  class. The book prints each as `class CommitMergeEpoch:  # continued from
  listing 3.9` so one rule can be read at a time.

## What is real and what is modeled

Every result in this chapter carries the label `simulated_decentralized`, and it
means something specific:

- **Real.** The arithmetic. Each worker's sum and count come from the kernel, the
  admission checks, close rules, freeze and merge are the chapter's protocol, and
  the committed mean is a real division over a real manifest.
- **Modeled.** Time and failure. Worker completion times are lognormal draws in
  units of the median worker, and dropout is an independent coin flip per worker
  per epoch. Nothing crosses a network.

No number here rests on a real multi-node deployment, and none of it should be
quoted as if it did.

## The one thing worth remembering

A quorum decides when an epoch may commit, not what the commit represents. The
committed mean is exact over its frozen manifest. Whether that manifest stands for
the workers that were left out depends on why they were late, which is what table
3.11 measures.
