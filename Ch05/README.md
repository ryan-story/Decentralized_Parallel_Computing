# Chapter 5: Preserving Ordered Progress with Missing Participants

The code for Chapter 5 of *Decentralized Parallel Programming*. It scans an ordered
sequence of 262,144 values split across 64 workers, first as an ordinary
distributed scan that needs every worker, then with Checkpointed Prefix Repair
(CPR), which accepts block summaries in any order, finalizes them in logical
order, and repairs a block whose worker has gone silent.

Section 5.2 of the book refers to this folder,
`Decentralized_Parallel_Computing/Ch05/`, as the chapter's companion code.

## The files, in the order the chapter builds them

| file | chapter section | listings | what it is |
|---|---|---|---|
| `kernels.cu` | 5.1.1, 5.1.2 | 5.1, 5.2 | `TileExclusiveScan`, the Blelloch exclusive scan of one tile in one CUDA block, and `ApplyCarryKernel`, which adds one carry to every element of a range. The only device code in the chapter. |
| `scan_launcher.py` | 5.1.2 | 5.3 | `scan_launcher`: scans a worker's logical block tile by tile, scans the tile totals by calling itself, and applies the tile carries. Returns the local scan, which stays on the device, and the block summary. |
| `finalize_block.py` | 5.3.1 | 5.4 | `finalize_block`: adds a block's carry to a copy of its local scan. |
| `distributed_scan.py` | 5.3.1 | 5.5 | The all-worker coordinator. Waits for every worker, computes the carries from the summaries in logical order, and finalizes every block. |
| `collective_scan.py` | 5.3.2 | 5.6 | The same carries from `MPI_Exscan`, one logical block per MPI rank. |
| `prefix_repair_state.py` | 5.4.1 | 5.7 | `SummaryCommit` and `PrefixRepairState`: the commit a worker sends and the coordinator's state. |
| `cpr_coordinator.py` | 5.4.2 | 5.8, 5.9 | `advance`, `receive` and `check_for_gap`: accept in any order, finalize in logical order, report the next persistent gap. |
| `cpr_scan.py` | 5.4.3 | 5.10, 5.11 | `cpr_worker`, `deliver` and `cpr_scan`: CPR on the wall clock, with spare GPUs for repairs. |
| `recover.py` | 5.4.4 | 5.12 | `recover`: rebuilds the coordinator from the checkpoint and the summary log. |
| `scan_experiments.py` | 5.1 to 5.5 | | The experiment harness: replays seeded arrival traces through the chapter's own `receive`, `advance`, `check_for_gap` and `recover`, and returns each result the chapter reports. |
| `experiment_harness.ipynb` | | | Runs the harness and prints the chapter's results: tables 5.1 to 5.7 and figure 5.3. Nothing else. |

Figures 5.1, 5.2 and 5.4 are diagrams, so none of them has a file.

## Getting started

```bash
pip install -r requirements.txt
```

The book's numbers were measured on an NVIDIA GPU (an RTX 5070). With an NVIDIA
device and CuPy installed, `kernels.cu` is compiled at run time and every block is
scanned there. Without one, a CPU fallback replays the same up-sweep and
down-sweep through NumPy in float32, in the same order, and gives the same values
bit for bit, so the notebook still reproduces the book's tables. Every entry point
prints which path it took. The one GPU-only result is table 5.2's measured kernel
time.

**Reproduce the book's numbers:**

```bash
jupyter lab experiment_harness.ipynb
```

It runs in under a minute on either path and reproduces every table value the
chapter quotes, along with figure 5.3. Every block is scanned by `scan_launcher`
and finalized by `finalize_block`, and every coordinator decision is made by the
chapter's functions; only the finish times, failures and replacement delays are
seeded. The kernel time in table 5.2 depends on the GPU, and varies from run to
run, so it will not match the book's RTX 5070 figures exactly.

**Run the listings directly:**

```bash
python scan_launcher.py                       # listing 5.3 on section 5.1.2's twelve values
python distributed_scan.py                    # listing 5.5
mpiexec -n 4 python collective_scan.py        # listing 5.6, four MPI ranks
python cpr_scan.py                            # listing 5.11, one worker fails, a spare repairs it
python recover.py                             # listing 5.12
```

## Running on fewer GPUs than the book assumes

Listings 5.5, 5.6 and 5.11 give each worker a device ID. Device IDs may repeat: on
a one-GPU machine, `[0] * 64` gives 64 workers that take turns on the same device,
and `cpr_scan` can use device 0 as its spare. The arithmetic is identical; only the
concurrency differs.

Listing 5.6 needs `mpi4py` and an MPI runtime. On Linux and macOS, install Open MPI
or MPICH and then `pip install mpi4py`. On Windows, `pip install mpi4py impi_rt`
installs Intel MPI with its `mpiexec`. All ranks may share one GPU.

## Where the code differs from the book's listings

The function and class bodies are the listings'. The differences are what it takes
to run them:

- `scan_launcher.py` falls back to NumPy when CuPy or a GPU is missing, and adds
  `gpu_available()` and `host_tile_scan()`, listing 5.1 replayed on the host.
- `finalize_block.py` applies the carry with NumPy when there is no device.
- Each listing's imports sit at the top of its module rather than inside the
  listing, and each module imports the earlier listings it calls.

The notebook's harness runs listing 5.11's coordination in simulated time rather
than on the wall clock: arrivals, gap deadlines and the coordinator restart of
table 5.7 are taken in time order, and each one is handed to the chapter's
`receive`, `check_for_gap` or `recover` exactly as the listing's loop would.

## What is real and what is modeled

Every result in this chapter carries the label `simulated_decentralized`, and it
means something specific:

- **Real.** The scans, the carries and the finalized output. Every summary comes
  from listing 5.3 and every finalized block from listing 5.4, both on the GPU, and
  every carry from listing 5.8. Each run is checked bitwise against listing 5.5 on the same blocks,
  and against a float64 scan of the whole input.
- **Modeled.** Time and failure. Finish times are lognormal draws, a failed worker
  never reports, a replacement worker reports a fixed 0.45 seconds after it is
  assigned, and the checkpoint and summary log live in memory. Nothing crosses a
  network.

No number here rests on a real multi-node deployment, and none of it should be
quoted as if it did.

## The one thing worth remembering

A worker can finish its local scan without its predecessors, but no block can be
finalized until every block before it is resolved. A quorum of workers cannot
stand in for a missing predecessor; the missing block has to be delivered or
recomputed, and CPR keeps every completed block ready until it is.
