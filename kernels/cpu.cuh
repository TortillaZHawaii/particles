#ifndef KERNELS_CPU_H
#define KERNELS_CPU_H

#include <GL/freeglut.h>
#include "../particles.cuh"

void h_colorBitmapFromParticles(GLubyte *bitmap, int width, int height, particles_t particles, int particles_count);
void h_steerParticles(particles_t particles, int particles_count, float dt, int screen_width, int screen_height);

#endif // KERNELS_GPU_H
