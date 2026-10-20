// Chapter 5, section 5.1: the scan kernels.
//
// A scan gives every position the accumulated state of everything before it.
// These are the only device code in the chapter.
//
//   TileExclusiveScan   listing 5.1   Blelloch exclusive scan of one tile in one
//                                     CUDA block: an up-sweep that stores subtree
//                                     sums, the tile total saved, the root seeded
//                                     with the identity, and a down-sweep that
//                                     passes prefix state to both children.
//                                     blockDim.x must be a power of two; n may be
//                                     anything up to blockDim.x.
//   ApplyCarryKernel    listing 5.2   adds one scalar carry to every element of a
//                                     range: equation 5.2 for prefix sum.
//
// The launcher in scan_launcher.py (listing 5.3) scans a worker's whole block by
// running TileExclusiveScan per tile, scanning the tile totals, and applying the
// tile carries with ApplyCarryKernel. finalize_block.py (listing 5.4) applies a
// block's carry with the same kernel.

extern "C" __global__ void TileExclusiveScan(
    const float* input, float* output, float* tile_total, int n) {
    extern __shared__ float temp[];
    int t = threadIdx.x;
    temp[t] = (t < n) ? input[t] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int i = (t + 1) * offset * 2 - 1;
        if (i < blockDim.x) temp[i] += temp[i - offset];
        __syncthreads();
    }

    if (t == 0) {
        *tile_total = temp[blockDim.x - 1];
        temp[blockDim.x - 1] = 0.0f;
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int i = (t + 1) * offset * 2 - 1;
        if (i < blockDim.x) {
            float left = temp[i - offset];
            temp[i - offset] = temp[i];
            temp[i] += left;
        }
        __syncthreads();
    }
    if (t < n) output[t] = temp[t];
}

extern "C" __global__ void ApplyCarryKernel(
    float* local_prefix,
    float carry,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        local_prefix[i] += carry;
    }
}
