# Chapter 4: Operating Under Variance, Skew, and Adversaries

The code for Chapter 4 of *Decentralized Parallel Programming*. It counts a skewed
stream of 200,000 events across sixteen workers, first as an ordinary distributed
histogram that needs every worker, then as an Influence-Limited Quorum Histogram
(ILQH), which commits once enough workers have contributed, rejects contributions
it can prove malformed, and bounds how far any one worker can move any one bin.

Section 4.2 of the book refers to this folder,
`Decentralized_Parallel_Programming/Ch04/`, as the chapter's companion code.

## The files, in the order the chapter builds them

| file | chapter section | listings | what it is |
|---|---|---|---|
| `kernels.cu` | 4.1.1, 4.1.3 | 4.1, 4.2 | `NaiveHistogramKernel`, which increments `h[b] += 1` with no synchronization and loses updates, and `PartialHistogramKernel`, the worker kernel: a block-private histogram in shared memory, `atomicAdd` inside the block, one published row per block. The only device code in the chapter. |
| `partial_histogram.py` | 4.1.3 | 4.3 | `partial_histogram`: supplies each block's shared memory, launches the worker kernel, and sums the published rows into one worker histogram. |
| `worker_contribution.py` | 4.3.1 | 4.4 | Binds one worker to one GPU and returns its histogram to the host as an int64 vector. |
| `distributed_histogram.py` | 4.3.2 | 4.5 | The all-worker coordinator. Runs every worker, checks each vector's shape, type and sign, adds it as it arrives, and returns once every shard has contributed. |
| `allreduce_histogram.py` | 4.3.3 | 4.6 | The same merge as a `torch.distributed` All-Reduce, so every rank ends up holding the global histogram. |
| `contribution.py` | 4.4.2 | 4.7, 4.8 | `HistogramContribution` and `validate_histogram`: the five checks a coordinator can make from a contribution and the epoch's declarations alone. |
| `bounded_merge.py` | 4.4.3 | 4.9 | `bounded_merge`: the exact merge with `tau=None`, and the influence limit of equation 4.3 with a `tau`. |
| `decentralized_histogram.py` | 4.4.4 | 4.10 | `histogram_worker` and `decentralized_histogram`: one epoch of ILQH on the wall clock. |
| `keystream.py` | 4.2.1 | | The reference key stream: 200,000 Zipf events over 16,384 keys, split by event across sixteen unequal batches, plus the variants the tables change. |
| `histogram_experiments.py` | 4.1 to 4.5 | | The experiment harness: drives `validate_histogram` and `bounded_merge` through seeded epochs and returns each result the chapter reports. |
| `experiment_harness.ipynb` | | | Runs the harness and draws the chapter's results: figures 4.2, 4.4, 4.5, 4.6 and 4.9, the coalition figure of section 4.5.2, and tables 4.1 to 4.9. Nothing else. |

Figures 4.1, 4.3, 4.7 (the coordinator flow) and 4.8 are diagrams, so none of them
has a file.

## Getting started

```bash
pip install -r requirements.txt
```

A GPU is optional. With an NVIDIA device and CuPy installed, `kernels.cu` is
compiled at run time and every worker's histogram is counted there; without one,
the same work runs on the CPU through NumPy and the counts are identical. Every
entry point prints which path it took. The race in figure 4.2 is the one exception:
it needs a GPU, because on the CPU there is nothing racing.

**Reproduce the book's numbers:**

```bash
jupyter lab experiment_harness.ipynb
```

It runs in under a minute on either path and reproduces every table value the
chapter quotes. Each worker's vector is computed by the chapter's kernel through
`worker_contribution`, contributions are admitted by `validate_histogram` and merged
by `bounded_merge`; only the completion times, dropout and adversaries are seeded.
Every experiment that runs the baseline timing uses the same 300 seeded epochs, so
a condition that appears in more than one table is one run and prints the same
numbers everywhere.

**Run the listings directly:**

```bash
python distributed_histogram.py     # listing 4.5
python allreduce_histogram.py       # listing 4.6, four processes over gloo
python decentralized_histogram.py   # listing 4.10, on the wall clock
```

## Running on fewer GPUs than the book assumes

Listings 4.5 and 4.10 give each worker a device ID. Device IDs may repeat: on a
one-GPU machine, `[0] * 16` gives sixteen workers that take turns on the same
device. The arithmetic is identical; only the concurrency differs.

Listing 4.6 needs PyTorch. `allreduce_histogram.py` joins four processes to one
process group over the gloo backend, which runs on the CPU on Linux, macOS and
Windows, then calls the listing on each. On a GPU cluster the same call runs over
NCCL; only the backend named in `init_process_group` changes.

The worker kernel keeps one int32 counter per bin in each block's shared memory, so
it can privatize a key universe only up to the device's per-block limit (25,344
bins on an RTX 5070, whose blocks may hold 101,376 bytes). Table 4.8's 32,768- and 65,536-bin rows are beyond
it, and their worker vectors are counted on the host, as section 4.5.3 says. The
two agree exactly wherever both run.

## Where the code differs from the book's listings

The function and class bodies are the listings'. The differences are what it takes
to run them:

- `partial_histogram.py` falls back to NumPy when CuPy or a GPU is missing, and
  adds three helpers: `gpu_available()`, `privatization_fits()`, and
  `naive_histogram()`, which launches listing 4.1 so the notebook can measure it.
- `worker_contribution.py` skips `cp.cuda.Device` when there is no device to bind.
- `allreduce_histogram.py` guards its PyTorch import, so the rest of the chapter
  imports without PyTorch.

The notebook's harness runs listing 4.10's epoch in simulated time rather than on
the wall clock: contributions arrive in order of their seeded completion times,
and the epoch closes at `alpha * t_k`, the deadline, or full participation, exactly
as the listing's loop does.

## What is real and what is modeled

Every result in this chapter carries the label `simulated_decentralized`, and it
means something specific:

- **Real.** The counting and the merge. Each worker's vector comes from the
  kernel, the admission checks and the influence limit are the chapter's code, and
  the committed histogram is a real sum over a real accepted set.
- **Modeled.** Time, failure and intent. Completion times are lognormal draws in
  units of the median worker, dropout is an independent coin flip per worker per
  epoch, and each Byzantine worker follows one fixed strategy: move every event it
  holds into one bin. Nothing crosses a network.

No number here rests on a real multi-node deployment, and none of it should be
quoted as if it did.

## The one thing worth remembering

Admission checks catch a contribution that is broken, not one that lies well. The
influence limit bounds how far a well-formed lie can move a bin, but only while
fewer than half the accepted workers collude, and it costs the honest fleet every
count in a bin that fewer than half of them observed.
