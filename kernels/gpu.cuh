#ifndef KERNELS_GPU_H
#define KERNELS_GPU_H

#include <GL/freeglut.h>
#include "../particles.cuh"

__global__ void d_colorBitmapFromParticles(GLubyte *bitmap, int width, int height, particles_t particles, int particles_count);
__global__ void d_steerParticles(particles_t particles, int particles_count, float dt, int screen_width, int screen_height);

#endif // KERNELS_GPU_H
