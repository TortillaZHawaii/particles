#include "shm.cuh"
#include "../cuda_utils.cuh"
#include "shared.cuh"

__device__ void dshm_prepareCharges(particles_t particles, int particles_count, float* charges, int id)
{
    int range = GET_BLOCKS_COUNT(particles_count);

    int start = id * range;
    int end = MIN(start + range, particles_count);

    for(int i = start; i < end; ++i)
    {
        charges[i] = particles.charges[i];
    }
}

__global__ void dshm_colorBitmapFromParticles(GLubyte *bitmap, int width, int height, particles_t particles, int particles_count)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float charges[];
    dshm_prepareCharges(particles, particles_count, charges, threadIdx.x);

    __syncthreads();

    colorBitmapFromParticles(id, particles, NULL, particles_count, bitmap, width, height);
}

__global__ void dshm_steerParticles(particles_t particles, int particles_count, float dt, int screen_width, int screen_height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float charges[];
    dshm_prepareCharges(particles, particles_count, charges, threadIdx.x);

    __syncthreads();

    steerParticle(id, particles, NULL, particles_count, dt, screen_width, screen_height);
}