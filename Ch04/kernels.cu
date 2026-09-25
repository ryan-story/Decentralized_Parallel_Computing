// Chapter 4, section 4.1: the histogram kernels.
//
// A histogram is a vector reduction: every event increments the counter of the
// bin its key names, and any thread's event may land in any bin, so threads
// contend for the same counters.
//
//   NaiveHistogramKernel    listing 4.1   h[b] += 1 with no synchronization.
//                                         Loses updates when two threads touch
//                                         the same bin; kept so the race can be
//                                         measured.
//   PartialHistogramKernel  listing 4.2   the worker kernel: a block-private
//                                         histogram in shared memory, atomicAdd
//                                         inside the block, and one published
//                                         row per block. Exact.
//
// Only PartialHistogramKernel is used by the rest of the chapter. The launcher in
// partial_histogram.py supplies its shared memory and sums the published rows
// into one worker histogram.

extern "C" __global__ void NaiveHistogramKernel(
    const int* keys,
    int* h,
    int n,
    int num_bins
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int b = keys[i];

        if (b >= 0 && b < num_bins) {
            h[b] += 1;
        }
    }
}


extern "C" __global__ void PartialHistogramKernel(
    const int* keys,
    int* block_hists,
    int n,
    int num_bins
) {
    extern __shared__ int s_h[];

    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        s_h[b] = 0;
    }
    __syncthreads();

    int stride = blockDim.x * gridDim.x;

    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n;
        i += stride
    ) {
        int b = keys[i];

        if (b >= 0 && b < num_bins) {
            atomicAdd(&s_h[b], 1);
        }
    }
    __syncthreads();

    int offset = blockIdx.x * num_bins;

    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        block_hists[offset + b] = s_h[b];
    }
}
