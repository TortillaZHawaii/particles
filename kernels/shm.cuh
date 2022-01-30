#ifndef KERNELS_SHM_H
#define KERNELS_SHM_H

#include <GL/freeglut.h>
#include "../particles.cuh"

#ifndef MIN
#define MIN(a, b) ((a < b) ? a : b)
#endif

__global__ void dshm_colorBitmapFromParticles(GLubyte *bitmap, int width, int height, particles_t particles, int particles_count);
__global__ void dshm_steerParticles(particles_t particles, int particles_count, float dt, int screen_width, int screen_height);

#endif // KERNELS_SHM_H