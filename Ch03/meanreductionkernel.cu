// Chapter 3, sections 3.1.1 and 3.1.2: the mean reduction kernels.
//
// A mean is reduced as the pair (sum, count). Both kernels run one reduction
// tree over one block, halving the number of active threads at every stage, and
// carry the count through the same tree as the sum.
//
//   MeanReductionKernel   listing 3.2   finalizes the mean inside the block
//   PartialMeanKernel     listing 3.3   returns (sum, count) and never divides
//
// Only PartialMeanKernel is launched by the rest of the chapter. A block that is
// one piece of a larger mean must not divide, or the coordinator ends up
// averaging means.
//
// Both kernels expect blockDim.x to be a power of two and v and count to hold
// exactly blockDim.x elements. The reduction happens in place in global memory,
// so v is overwritten.
//
// extern "C" keeps the names unmangled so CuPy can look the kernels up by name.

extern "C" __global__ void MeanReductionKernel(float* v, unsigned int* count, float* reduction) {

    unsigned int i = threadIdx.x;

    count[i] = 1;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            v[i] += v[i + stride];
            count[i] += count[i + stride];
        }

    __syncthreads();
    }
    if (threadIdx.x == 0) {
        *reduction = v[0] / count[0];
    }
}


extern "C" __global__ void PartialMeanKernel(float* v, unsigned int* count,
                                             float* out_sum, unsigned int* out_count) {

    unsigned int i = threadIdx.x;

    count[i] = 1;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            v[i] += v[i + stride];
            count[i] += count[i + stride];
        }

    __syncthreads();
    }
    if (threadIdx.x == 0) {
        *out_sum = v[0];
        *out_count = count[0];
    }
}
