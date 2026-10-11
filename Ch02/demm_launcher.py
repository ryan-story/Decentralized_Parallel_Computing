"""Chapter 2, section 2.3.2: the DeMM worker.

    orth(X)                                           orthonormalize the columns of X
    demm_launcher(device_id, A_r, B_r, q, s, h,       listing 2.8, Algorithm 2.1.
                  _SRC, seed)                         Returns the low-rank factors
                                                      L_r and M_r^T whose product
                                                      approximates C_r = A_r B_r.

The partial product C_r is never formed. Every multiplication is MatrixMulKernel
applied to a thin operand through `matrix_mul`: probe, refine, project, then a
thin SVD of the small projected matrix K_r.

The function bodies are the listing's; `cp` comes from `cupy_or_numpy`.
"""

from __future__ import annotations

from cupy_or_numpy import cp
from matmul_launcher import matrix_mul


def orth(X):
    Q, _ = cp.linalg.qr(X, mode="reduced")
    return Q


def demm_launcher(device_id, A_r, B_r, q, s, h, _SRC, seed):
    with cp.cuda.Device(device_id):
        _MODULE = cp.RawModule(code=_SRC, options=("--std=c++14",))
        _matmul = _MODULE.get_function("MatrixMulKernel")

        A_r = cp.asarray(A_r, dtype=cp.float32, order="C")
        B_r = cp.asarray(B_r, dtype=cp.float32, order="C")

        m, p_r = A_r.shape
        n = B_r.shape[1]

        # Set the sketch width.
        ell = min(q + s, m, n, p_r)

        # MatrixMulKernel requires contiguous row-major inputs.
        A_r_T = cp.ascontiguousarray(A_r.T)
        B_r_T = cp.ascontiguousarray(B_r.T)

        # Generate the random probe matrix.
        random = cp.random.RandomState(seed)
        Omega_r = random.standard_normal((n, ell), dtype=cp.float32)

        # Apply the forward operator:
        # Y_r = A_r * (B_r * Omega_r)
        T_r = matrix_mul(B_r, Omega_r, _matmul)
        Y_r = matrix_mul(A_r, T_r, _matmul)
        Q_r = orth(Y_r)

        # Refine the sampled subspaces.
        for _ in range(h):
            # Apply the transpose operator:
            # Z_r = B_r.T * (A_r.T * Q_r)
            S_r = matrix_mul(A_r_T, Q_r, _matmul)
            Z_r = matrix_mul(B_r_T, S_r, _matmul)
            P_r = orth(Z_r)

            # Apply the forward operator:
            # Y_r = A_r * (B_r * P_r)
            T_r = matrix_mul(B_r, P_r, _matmul)
            Y_r = matrix_mul(A_r, T_r, _matmul)
            Q_r = orth(Y_r)

        # Project the partial product:
        # K_r = (Q_r.T * A_r) * B_r
        Q_r_T = cp.ascontiguousarray(Q_r.T)
        W_r = matrix_mul(Q_r_T, A_r, _matmul)
        K_r = matrix_mul(W_r, B_r, _matmul)

        # Compute the thin SVD of K_r.
        U_hat_r, singular_values, V_r_T = cp.linalg.svd(
            K_r,
            full_matrices=False,
        )

        # Keep the leading q components.
        U_hat_r_q = U_hat_r[:, :q]
        V_r_q_T = V_r_T[:q, :]

        # Lift the left singular vectors.
        U_r_q = matrix_mul(Q_r, U_hat_r_q, _matmul)

        # Divide the singular values between the factors.
        singular_scale = cp.sqrt(singular_values[:q])

        L_r = U_r_q * singular_scale[None, :]
        M_r_T = singular_scale[:, None] * V_r_q_T

        return cp.asnumpy(L_r), cp.asnumpy(M_r_T)
