#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define BLOCK_SIZE 512

#ifndef GET_BLOCKS_COUNT
#define GET_BLOCKS_COUNT(value) value / BLOCK_SIZE + (value % BLOCK_SIZE ? 1 : 0)
#endif // GET_BLOCKS_COUNT

#endif // CUDA_UTILS_H