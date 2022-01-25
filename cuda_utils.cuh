#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define BLOCK_SIZE 1024

__device__ __host__ uint getBlocksCount(uint value)
{
    return value / BLOCK_SIZE + (value % BLOCK_SIZE ? 1 : 0);
}

#endif // CUDA_UTILS_H