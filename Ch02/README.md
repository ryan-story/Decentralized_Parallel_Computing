# Chapter 2: Your First Decentralized Parallel System

The code for Chapter 2 of *Decentralized Parallel Programming*. It builds an
item-item recommender whose users are split across workers and never pooled, two
ways: exactly, and with DeMM.

Everything here is small on purpose. There is one CUDA kernel, and every later
file is a different way of calling it.

## The files, in the order the chapter builds them

| file | chapter section | what it is |
|---|---|---|
| `matmulkernel.cu` | 2.2.3 | `MatrixMulKernel`, one thread per output element. The only device code in the chapter. |
| `matmul_launcher.py` | 2.3.1 | Blocked matmul. `matmul_launcher` is one worker's whole job; `matrix_mul` is the same launch with the result left on the device so it can feed another multiply. |
| `distributed_gram_coordinator.py` | 2.3.2 | The exact baseline. Shards users, has each worker return a full partial `R_r^T R_r`, adds them. Correct, and expensive. |
| `demm_launcher.py` | 2.4.2 | The DeMM worker: Algorithm 2.1 with the Gram specialization applied, returning a rank-`q` factor `L_r` and the exact item-count diagonal. Never forms the partial product. |
| `demm_gram_coordinator.py` | 2.4.2 | The DeMM coordinator. Same shape as the exact one; workers just return less. Also serves recommendations straight from the factors. |
| `movielens.py` | plumbing | Data loading and the held-out recall metric, shared by the notebook and the app so they cannot disagree. |
| `experiment_harness.ipynb` | 2.1 | The experiment the book reports: Table 2.1 and Figure 2.2, and nothing else. |
| `app/` | | The recommender, running, on localhost. |

## Getting started

```bash
pip install -r requirements.txt
```

A GPU is optional. With an NVIDIA device and CuPy installed, `matmulkernel.cu` is
compiled at run time and the multiplies run there; without one, the identical code
runs on the CPU through NumPy. Both the notebook and the app print which path they
took, because a claim about a decentralized system should never be vague about
what actually ran.

**Run the recommender:**

```bash
cd app && ./app_launch.sh          # then open http://localhost:8501
```

Pick a few films, and the two paths answer side by side: nearly the same
recommendations, about twenty times apart in bytes on the small dataset and about
a hundred and twenty times apart at the scale the book reports. Uncheck a worker
in the sidebar to watch the merge lose exactly that worker's term and nothing else.

**Reproduce the book's numbers:**

```bash
jupyter lab experiment_harness.ipynb
```

It defaults to `ml-latest-small` and finishes in seconds. Set
`DATASET = "ml-25m"` in the environment cell for the configuration the book
reports: 162,539 users over the 4,000 most-rated items, where the exact exchange
is 512 MB and the rank-32 DeMM summary is 4.22 MB.

## What is real and what is modeled

Every result in this chapter carries the label `simulated_decentralized`, and it
means something specific:

- **Real.** The arithmetic. The kernel runs on a real GPU when one is present, the
  merge is a real sum, and the recommendations are real recommendations from real
  MovieLens data.
- **Modeled.** The network. The workers are threads on one machine. We count the
  bytes each summary would cross a wide-area link and price them at a stated
  bandwidth; we do not send them between machines.

No number here rests on a real multi-node deployment, and none of it should be
quoted as if it did. The app says so on screen for the same reason.

## The one thing worth remembering

The exact baseline sends an `n_items` by `n_items` matrix per worker. DeMM sends
an `n_items` by `q` factor and a diagonal. Neither depends on how many users a
worker holds, which is why the gap widens as the system grows: local work scales
with users, and the exchange does not.
