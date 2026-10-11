// Chapter 2, sections 2.1.1 to 2.1.3: the matrix multiplication kernel, one thread
// per output element (listing 2.4).
//
// This is the only device code in the chapter. Everything after it is a different
// way of calling it:
//
//   section 2.2  blocked matmul   one launch per worker, on the slabs A(I) and B(J)
//   section 2.3  DeMM             several launches per worker, on thin operands,
//                                 so the partial product is never formed
//
// Row-major indexing throughout, as equation 2.3 derives: A[i, k] == a[i * p + k].

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
