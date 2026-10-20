// Chapter 6: the convolution kernels. The only device code in the chapter.
//
//   Conv2DBasicKernel   listing 6.1   one thread per output; the filter and every
//                                     input value read from global memory; zero
//                                     padding outside the image.
//   Conv2DConstKernel   listing 6.3   the same, with the filter in constant memory.
//   Conv2DTiledKernel   listing 6.4   the block stages its input tile and the
//                                     tile's halo in shared memory, then computes
//                                     from there.
//   ConvLayerKernel     listing 6.6   one convolution layer of the network: C input
//                                     channels, 3 x 3 filters, bias and ReLU, over
//                                     an already padded input region.
//
// Compiled at run time by NVRTC through CuPy's RawModule.

extern "C" __global__ void Conv2DBasicKernel(
    const float *N, const float *F, float *P, int height, int width, int r
) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= height || out_x >= width) return;
    int fw = 2 * r + 1;
    float acc = 0.0f;
    for (int fy = -r; fy <= r; ++fy) {
        for (int fx = -r; fx <= r; ++fx) {
            int in_y = out_y + fy, in_x = out_x + fx;
            if (0 <= in_y && in_y < height && 0 <= in_x && in_x < width)
                acc += N[in_y * width + in_x] * F[(fy + r) * fw + fx + r];
        }
    }
    P[out_y * width + out_x] = acc;
}

#define MAX_RADIUS 4
__constant__ float FILTER[(2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1)];

extern "C" __global__ void Conv2DConstKernel(
    const float *N, float *P, int height, int width, int r
) {
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= height || out_x >= width) return;
    int fw = 2 * r + 1;
    float acc = 0.0f;
    for (int fy = -r; fy <= r; ++fy) {
        for (int fx = -r; fx <= r; ++fx) {
            int in_y = out_y + fy, in_x = out_x + fx;
            if (0 <= in_y && in_y < height && 0 <= in_x && in_x < width)
                acc += N[in_y * width + in_x] * FILTER[(fy + r) * fw + fx + r];
        }
    }
    P[out_y * width + out_x] = acc;
}

extern "C" __global__ void Conv2DTiledKernel(
    const float *N, float *P, int height, int width, int r
) {
    extern __shared__ float tile[];
    int tile_w = blockDim.x + 2 * r, tile_h = blockDim.y + 2 * r;
    int y0 = blockIdx.y * blockDim.y - r, x0 = blockIdx.x * blockDim.x - r;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int t = tid; t < tile_w * tile_h; t += blockDim.x * blockDim.y) {
        int in_y = y0 + t / tile_w, in_x = x0 + t % tile_w;
        bool inside = 0 <= in_y && in_y < height && 0 <= in_x && in_x < width;
        tile[t] = inside ? N[in_y * width + in_x] : 0.0f;
    }
    __syncthreads();
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= height || out_x >= width) return;
    int fw = 2 * r + 1;
    float acc = 0.0f;
    for (int fy = 0; fy < fw; ++fy)
        for (int fx = 0; fx < fw; ++fx)
            acc += tile[(threadIdx.y + fy) * tile_w + threadIdx.x + fx]
                 * FILTER[fy * fw + fx];
    P[out_y * width + out_x] = acc;
}

extern "C" __global__ void ConvLayerKernel(
    const float *X, const float *Wt, const float *bias, float *Y,
    int C, int H, int W
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.z;
    if (x >= W || y >= H) return;
    int in_h = H + 2, in_w = W + 2;
    float acc = bias[m];
    for (int c = 0; c < C; ++c) {
        const float *xc = X + (size_t)c * in_h * in_w;
        const float *wc = Wt + ((size_t)m * C + c) * 9;
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q)
                acc += xc[(y + p) * in_w + x + q] * wc[p * 3 + q];
    }
    Y[((size_t)m * H + y) * W + x] = fmaxf(acc, 0.0f);
}
