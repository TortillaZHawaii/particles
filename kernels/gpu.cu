#include "gpu.cuh"
#include "shared.cuh"

__global__ void d_colorBitmapFromParticles(GLubyte *bitmap, int width, int height, particles_t particles, int particles_count)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    colorBitmapFromParticles(id, particles, NULL, particles_count, bitmap, width, height);
}

__global__ void d_steerParticles(particles_t particles, int particles_count, float dt, int screen_width, int screen_height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    steerParticle(id, particles, NULL, particles_count, dt, screen_width, screen_height);
}
