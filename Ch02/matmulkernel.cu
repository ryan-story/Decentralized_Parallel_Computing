// Chapter 2, section 2.2.3: the ordinary matrix multiply, one thread per output
// element.
//
// This is the only device kernel the chapter needs. Everything after it is a
// different way of CALLING it:
//
//   section 2.3.1  blocked matmul      one call per worker, on a slice of the
//                                      contracted dimension
//   section 2.3.2  distributed Gram    the same, with A = R^T and B = R
//   section 2.4.2  DeMM                several calls per worker, on thin operands,
//                                      so the partial product is never formed
//
// The operands are rectangular rather than square because the blocked and Gram
// cases need that: a worker holding a slice of the contraction multiplies an
// [m, p] block by a [p, n] block, and those are only square by coincidence.
//
// Row-major indexing throughout, as section 2.2.3 derives in equations 2.4 and
// 2.5: A[i, j] == A[i * p + j], so the stride is the matrix's own column count.

extern "C" __global__ void MatrixMulKernel(
    const float* A,      // [m, p] left operand
    const float* B,      // [p, n] right operand
    float* C,            // [m, n] output
    int m,               // output matrix row dimension
    int n,               // output matrix col dimension
    int p                // shared inner dimension: A columns and B rows
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {   // boundary guard
        float summation = 0.0f;
        for (int k = 0; k <= p - 1; ++k) {
            summation += A[row * p + k] * B[k * n + col];
        }
        C[row * n + col] = summation;
    }
}
