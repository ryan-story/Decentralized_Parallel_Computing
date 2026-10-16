# Chapter 2: Your First Decentralized Parallel System

The code for Chapter 2 of *Decentralized Parallel Programming*. It builds a matrix
multiplication kernel, runs it across workers two ways, as blocked matrix
multiplication and as decentralized matrix multiplication (DeMM), and uses both to
build an item-item recommender whose users are split across workers and never
pooled.

Section 2.4.2 of the book refers to this folder,
`Decentralized_Parallel_Computing/Ch02/`, as the chapter's companion code.

## The files, in the order the chapter builds them

| file | chapter section | listings | what it is |
|---|---|---|---|
| `matmulkernel.cu` | 2.1.1 to 2.1.3 | 2.4 | `MatrixMulKernel`, one thread per output element. The only device code in the chapter. |
| `matmul_launcher.py` | 2.2.2, 2.3.2 | 2.6 | `matmul_launcher` launches the kernel on one worker's device and returns the result to the host; `matrix_mul` is the same launch with the result left on the device, so it can feed another multiplication. |
| `blocked_matmul_coordinator.py` | 2.2.2, 2.4 | 2.7 | `blocked_matmul` cuts the output into blocks C(I, J) and gives each worker the slabs A(I) and B(J). `blocked_gram` applies it to the item-item model, exactly. |
| `demm_launcher.py` | 2.3.2 | 2.8 | `orth` and `demm_launcher`: Algorithm 2.1, returning the low-rank factors L_r and M_r^T without ever forming the partial product. |
| `demm_coordinator.py` | 2.3.2 | 2.9 | `demm_matmul` splits the shared dimension across workers and sums their approximate partial products. |
| `demm_gram_coordinator.py` | 2.4.1 | | DeMM applied to the item-item model: each worker sends one factor, because its partial product is symmetric, together with its exact item counts. Also serves recommendations straight from the factors. |
| `cupy_or_numpy.py` | | | CuPy on a machine with a CUDA device, a NumPy stand-in otherwise, so the same code runs on either. |
| `movielens.py` | 2.4 | | Data loading and the held-out recall metric, shared by the notebook and the app so they cannot disagree. |
| `experiment_harness.ipynb` | 2.4.3 | | The experiment the book reports: Table 2.1 and Figure 2.13, and nothing else. |
| `app/` | 2.4.2 | | The recommender of figure 2.12, running on localhost. |

Listings 2.1 to 2.3 build the kernel up step by step, and listings 2.2 and 2.5 show
it inline in a Python string; `matmulkernel.cu` is where it ends up. The book writes
listings 2.7 and 2.9 as short programs; here the same statements are the bodies of
`blocked_matmul` and `demm_matmul`, so the notebook and the app can import them.

## Getting started

```bash
pip install -r requirements.txt
```

A GPU is optional. With an NVIDIA device and CuPy installed, `matmulkernel.cu` is
compiled at run time and the multiplications run there; without one, the same code
runs on the CPU through NumPy. Both the notebook and the app print which path they
took, because a claim about a decentralized system should never be vague about what
actually ran.

**Run the recommender:**

```bash
cd app && ./app_launch.sh          # then open http://localhost:8501
```

Choose the small or full dataset, pick a few films, and the two methods answer side
by side, with the bytes each one moves. Change the number of workers, adjust the
DeMM rank, or uncheck a worker in the sidebar to see what a missing worker costs
each method.

**Reproduce the book's numbers:**

```bash
jupyter lab experiment_harness.ipynb
```

It runs the configuration the book reports: MovieLens 25M, 162,539 users over the
4,000 most-rated items, eight workers. The first run downloads the dataset (about
250 MB); after that the notebook takes about a minute on a GPU. For a quick pass,
switch the environment cell to `ml-latest-small`.

**Run the listings directly:**

```bash
python matmul_launcher.py            # listing 2.6 against a centralized product
python blocked_matmul_coordinator.py # listing 2.7
python demm_coordinator.py           # listing 2.9, error against rank
```

## What is real and what is modeled

Every result in this chapter carries the label `simulated_decentralized`, and it
means something specific:

- **Real.** The arithmetic. The kernel runs on a real GPU when one is present, the
  merge is a real sum, and the recommendations are real recommendations from real
  MovieLens data.
- **Modeled.** The network. The workers are threads on one machine. The byte counts
  are what each method would send across a wide-area link; nothing is sent between
  machines.

No number here rests on a real multi-node deployment, and none of it should be
quoted as if it did.
