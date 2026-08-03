"""Chapter 2, section 2.4.2: the DeMM worker.

This implements Algorithm 2.2, the DeMM Gram summary, which is Algorithm 2.1 with
the two specializations of section 2.4.1 applied.

Algorithm 2.1 in general produces two factors, `L_r` and `M_r^T`, whose product
approximates the partial product `C_r = A_r B_r`. For the item-item model the
partial product is `R_r^T R_r`, which is symmetric and positive semi-definite, so
its left and right singular subspaces are the same subspace and `M_r^T = L_r^T`.
The second factor carries nothing the first does not. A worker therefore sends
`L_r` alone, and the coordinator forms `L_r L_r^T`. Measured on MovieLens 25M at
rank 32 the two forms reconstruct the model equally well, 0.042 against 0.040
relative off-diagonal error, but two factors cost 8.19 MB where one costs 4.22 MB.

The worker also returns `diag_r`, the exact item-count diagonal, which is the
second half of section 2.4.1's specialization. Low-rank truncation shrinks that
diagonal, and because it sits in a denominator the error is not a small
perturbation of a score but a systematic distortion of every similarity. It costs
one number per item, so 4,000 items is 16 KB, and dropping it costs recall@50
0.386 against 0.277. That failure does not show up on a small dataset; it appears
at the scale the chapter reports, which is a good reason to distrust a small-data
check of a normalizer.

The two specializations have different reach, and conflating them is how the
diagonal gets dropped by someone adapting this file. SYMMETRY is a property of the
Gram: returning one factor instead of two is only valid because `R_r^T R_r` is
symmetric, and outside a Gram the second factor is genuinely needed. THE DIAGONAL
IS A PROPERTY OF THE CONSUMER: it is required because the recommender divides by
`diag(G)`, so any DeMM whose consumer normalizes that way needs it whether it
ships one factor or two. Measured on the same data, the general two-factor form
without the diagonal reaches recall@50 0.277 against 0.384 with it, the same
collapse. If you reuse Algorithm 2.1 for a product that is not a Gram, drop the
symmetry and keep the habit: send exactly the quantities the consumer divides by.

The partial product `C_r` is never formed. The worker only ever applies `R_r` and
its transpose to THIN matrices, using `MatrixMulKernel` several times over.

Deviations from the book's listing, and why:

  * `_SRC` is gone: the kernel is compiled once at import, in `matmul_launcher`.
  * `_device` and `to_host` move arrays between host and device, so the same
    function body runs with or without a GPU. The book, writing for a machine that
    has one, uses `cp` directly.
  * The signature, the algorithm, the order of operations and the returned
    `(L_r, diag_r)` are the book's.
"""

from __future__ import annotations

import numpy as np

from matmul_launcher import matrix_mul, to_host

try:
    import cupy as cp
except Exception:                                 # pragma: no cover
    cp = None


def orth(X):
    """Orthonormalize the columns of X: `orth` in Algorithm 2.1.

    Runs on the host through NumPy's QR. The matrices reaching it are thin,
    `n_items` by `ell` with `ell` a few dozen, so the factorization is cheap and
    the transfer is small; everything that scales with the number of users stays
    on the device in `matrix_mul`.
    """
    Q, _ = np.linalg.qr(np.asarray(to_host(X), dtype=np.float64), mode="reduced")
    return _device(Q)


def _device(X):
    """Put a host array back where `matrix_mul` expects to find it."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    return cp.asarray(X) if (cp is not None and cp.cuda.runtime.getDeviceCount() > 0) else X


def demm_launcher(device_id, A_r, B_r, q, s=8, h=1, seed=0):
    """One worker's DeMM summary, section 2.4.2.

    Takes the same blocks the blocked-matmul worker takes, `A_r` and `B_r`, so the
    coordinator hands both workers their slices the same way. For the Gram those
    are `A_r = R_r^T` and `B_r = R_r`, which is why the exact diagonal can be read
    straight off `B_r`.

    Returns `(L_r, diag_r)`:
        L_r     [n_items, q]   the low-rank factor, with L_r L_r^T ~ R_r^T R_r
        diag_r  [n_items]      the exact item-count diagonal

    `q` is the target rank, `s` the oversampling, and `h` the number of refinement
    iterations, exactly as in Algorithm 2.1.
    """
    A_r = np.ascontiguousarray(np.asarray(A_r, dtype=np.float32))
    B_r = np.ascontiguousarray(np.asarray(B_r, dtype=np.float32))
    m, p_r = A_r.shape
    n = B_r.shape[1]

    ell = int(min(q + s, m, n, p_r))
    q = int(min(q, ell))

    def _run():
        A_r_d = _device(A_r)                     # [m, p_r]
        B_r_d = _device(B_r)                     # [p_r, n]
        A_r_T = _device(A_r.T)
        B_r_T = _device(B_r.T)

        # Generate the random probe matrix.
        rng = np.random.default_rng(seed)
        Omega_r = _device(rng.standard_normal((n, ell)))

        # Apply the forward operator: Y_r = A_r (B_r Omega_r).
        T_r = matrix_mul(B_r_d, Omega_r)
        Y_r = matrix_mul(A_r_d, T_r)
        Q_r = orth(Y_r)

        # Refine the sampled subspaces.
        for _ in range(int(h)):
            # transpose operator: Z_r = B_r^T (A_r^T Q_r)
            S_r = matrix_mul(A_r_T, Q_r)
            Z_r = matrix_mul(B_r_T, S_r)
            P_r = orth(Z_r)
            # forward operator: Y_r = A_r (B_r P_r)
            T_r = matrix_mul(B_r_d, P_r)
            Y_r = matrix_mul(A_r_d, T_r)
            Q_r = orth(Y_r)

        # Project the partial product: K_r = (Q_r^T A_r) B_r.
        W_r = matrix_mul(_device(to_host(Q_r).T), A_r_d)
        K_r = matrix_mul(W_r, B_r_d)

        # Thin SVD of the small projected matrix, on the host.
        U_hat_r, singular_values, _ = np.linalg.svd(
            np.asarray(to_host(K_r), dtype=np.float64), full_matrices=False)

        # Keep the leading q components and lift them back into the item space.
        U_hat_r_q = _device(U_hat_r[:, :q])
        U_r_q = to_host(matrix_mul(Q_r, U_hat_r_q))

        # L_r is exactly what the general case builds: U_r,q Sigma^(1/2). The
        # square root stays even though there is only one factor now, because L_r
        # appears twice in L_r L_r^T. Using Sigma here instead of its root would
        # reconstruct Sigma^2 and the error is catastrophic, not subtle.
        scale = np.sqrt(np.maximum(singular_values[:q], 0.0)).astype(np.float32)
        return (U_r_q * scale[None, :]).astype(np.float32)

    if cp is not None and cp.cuda.runtime.getDeviceCount() > 0:
        with cp.cuda.Device(device_id):
            L_r = _run()
    else:
        L_r = _run()

    # The exact item-count diagonal, read straight off B_r as the book's listing
    # does: for the Gram, B_r is R_r, so this is diag(R_r^T R_r).
    diag_r = (B_r * B_r).sum(axis=0).astype(np.float32)
    return L_r, diag_r
