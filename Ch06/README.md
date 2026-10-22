# Chapter 6: Independence Geometry

The code for Chapter 6 of *Decentralized Parallel Programming*. It builds convolution
on one GPU, extends it to a trained convolutional network, splits the network across
16 workers that exchange halos, and serves it with Versioned Halo Completion (VHC):
every output whose inputs are known is computed, every halo cell is sent as soon as it
is exact, and a silent worker's region goes to a spare.

Section 6.2 of the book refers to this folder,
`Decentralized_Parallel_Computing/Ch06/`, as the chapter's companion code.

## The files, in the order the chapter builds them

| file | chapter section | listings | what it is |
|---|---|---|---|
| `kernels.cu` | 6.1.1, 6.1.3 | 6.1, 6.3, 6.4, 6.6 | `Conv2DBasicKernel`, `Conv2DConstKernel` (filter in constant memory), `Conv2DTiledKernel` (tile and halo in shared memory), and `ConvLayerKernel`, one layer of the network over a padded input region. The only device code in the chapter. |
| `image_convolution.py` | 6.1.1 | 6.2, 6.5 | Launch the three image kernels. Its demo runs the worked example of section 6.1. |
| `cnn.py` | 6.1.3 | 6.7, 6.8 | `conv_layer`, `max_pool`, `head` and `cnn_forward`: the whole network on one GPU, and `load_weights()`. |
| `distributed_cnn.py` | 6.3.1 | 6.9, 6.10 | `pad_with_halos` and `distributed_cnn`: the network across a 4 x 4 grid of workers, exchanging halos before every convolution. |
| `collective_cnn.py` | 6.3.2 | 6.11 | The same exchange with MPI, one tile per rank, in two phases so the corners arrive. Run with `mpiexec -n 16`. |
| `halo_commit.py` | 6.4.1 | 6.12, 6.13 | `HaloCommit`, `TileState`, `band` and `open_tile`: VHC's message and a tile's state. |
| `vhc_tile.py` | 6.4.2 | 6.14, 6.15, 6.16 | `receive_halo`, `advance`, `carry` and `ready_strips`: exact progress on one tile, cell by cell. |
| `vhc_coordinator.py` | 6.4.3 | 6.17 | `CompletionState`, `heard`, `check_for_missing` and `accept_final`. |
| `vhc_infer.py` | 6.4.3 | 6.18, 6.19 | `vhc_worker`, `final_logits` and `vhc_infer`: VHC on the wall clock, one thread per tile, with a spare taking over a silent tile. |
| `model.py` | 6.1.3 | | The network's definition, the data download and preprocessing, the PyTorch reference, and the training that produced the weights. |
| `conv_experiments.py` | 6.1, 6.2, 6.5 | | The experiment harness: runs the chapter's listings and returns each result the chapter reports. |
| `experiment_harness.ipynb` | | | Runs the harness and prints the chapter's experiment results: tables 6.1, 6.4 and 6.6 to 6.12, figures 6.3, 6.4 and 6.6, and the exactness counts. Nothing else. |
| `model/weights.npz` | | | The frozen weights every number in the book was measured with. |
| `model/costs.json` | | | Listing 6.6's kernel cost per layer, measured on the book's RTX 5070, which the harness uses as worker compute times. |
| `assets/chelsea.png` | 6.1.3 | | The photograph in figure 6.3, by Stefan van der Walt, released under CC0 and distributed with scikit-image. |

Figures 6.1, 6.2 and 6.5 are diagrams, and tables 6.2, 6.3 and 6.5 hold definitions and
settings, so none of these is generated here.

## Getting started

```bash
pip install -r requirements.txt
```

The chapter needs an NVIDIA GPU with CuPy: every convolution runs through
`kernels.cu`, compiled at run time, and the exactness results compare GPU outputs bit
for bit. There is no CPU fallback. The book's numbers were measured on an RTX 5070.

The first run downloads Imagenette (fast.ai's 10-class subset of ImageNet, 160 px
version, about 94 MB) into `data/` and decodes it once into a cache there. `data/` is
never committed.

**Reproduce the book's numbers:**

```bash
jupyter lab experiment_harness.ipynb
```

Every simulated result (tables 6.6 to 6.12, figures 6.4 and 6.6) reproduces exactly:
the arithmetic runs on your GPU, but worker compute times come from
`model/costs.json`, not from your GPU, and every draw is seeded. Tables 6.1 and 6.4
time your GPU, so they vary from run to run and will not match the book's RTX 5070
figures. The substituted-halo table takes about an hour, and the exactness check over
all 3,925 validation images over an hour; the notebook says how to shorten the latter.

**Run the listings directly:**

```bash
python image_convolution.py                  # the worked example, and the three image kernels
python cnn.py                                # classify eight validation images
python distributed_cnn.py                    # listing 6.10, bitwise against one GPU
mpiexec -n 16 python collective_cnn.py       # listing 6.11, 16 MPI ranks
python vhc_infer.py                          # listing 6.19, one tile's worker lost, a spare repairs it
python model.py                              # retrain into model/retrained.npz (about 4 minutes)
```

## Running on fewer GPUs than the book assumes

Listings 6.10 and 6.19 give each worker a device ID, and IDs may repeat: `[0] * 16`
gives 16 workers that share one GPU. The arithmetic is identical; only the
concurrency differs.

Listing 6.11 needs `mpi4py` and an MPI runtime. On Linux and macOS, install Open MPI
or MPICH and then `pip install mpi4py`. On Windows, `pip install mpi4py impi_rt`
installs Intel MPI with its `mpiexec`. All 16 ranks may share one GPU.

## Where the code differs from the book's listings

The function and class bodies are the listings'. The differences are what it takes
to run them:

- Each listing's imports sit at the top of its module, and each module imports the
  earlier listings it calls.
- `cnn.py` adds `load_weights()`, and `model.py` holds the data handling and training
  that the chapter describes in prose.

The notebook's harness runs VHC in simulated time rather than on the wall clock:
deliveries, heartbeats and repair checks are taken in time order, and each one is
handed to the chapter's `receive_halo`, `heard`, `check_for_missing` and
`accept_final` exactly as listing 6.19's loop would. The conventional exchange is
modeled with the best schedule listing 6.10 permits: each worker waits only for its
own eight neighbors and computes its interior while it waits.

## What is real and what is modeled

Every result in this chapter carries the label `simulated_decentralized`, and it
means something specific:

- **Real.** Every convolution, pool and head, run on the GPU through the chapter's
  listings, and every VHC decision. Each completed request is checked bitwise
  against listing 6.8 on one GPU.
- **Modeled.** Time and failure. Delivery times are a link's one-way latency with
  seeded jitter plus bytes over bandwidth; worker compute times are the measured
  kernel costs scaled by a seeded speed per worker; a failed worker never responds;
  a spare starts a fixed 0.45 seconds after it is assigned. Nothing crosses a
  network.

No number here rests on a real multi-node deployment, and none of it should be
quoted as if it did.

## The one thing worth remembering

An output depends only on a known window of inputs, so a missing worker cannot affect
the outputs whose windows avoid its region. A protocol that advances layer by layer
loses all of them; one that computes and sends each cell as soon as its window is
known keeps every one, and only the missing region has to wait for repair.
