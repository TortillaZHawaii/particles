#ifndef KERNELS_SHARED_H
#define KERNELS_SHARED_H

#include <GL/freeglut.h>
#include <helper_math.h>
#include "../particles.cuh"

__device__ __host__ void steerParticle(int id, particles_t particles, float* charges,
    int particles_count, float dt, int screen_width, int screen_height);
__device__ __host__ void colorBitmapFromParticles(int id, particles_t particles, float* charges,
    int particles_count, GLubyte *bitmap, int width, int height);

#endif // KERNELS_SHARED_H
