#ifndef SETUP_H
#define SETUP_H

#include "particles.cuh"

void allocateParticlesOnHost(particles_t *h_particles, int particles_count)
{
    ssize_t size = sizeof(float) * particles_count;

    h_particles->positions_x = (float*)malloc(size);
    h_particles->positions_y = (float*)malloc(size);

    h_particles->velocities_x = (float*)malloc(size);
    h_particles->velocities_y = (float*)malloc(size);

    h_particles->charges = (float*)malloc(size);
    h_particles->masses = (float*)malloc(size);

    // check for malloc errors
    if (h_particles->positions_x == NULL || h_particles->positions_y == NULL ||
        h_particles->velocities_x == NULL || h_particles->velocities_y == NULL ||
        h_particles->charges == NULL || h_particles->masses == NULL)
    {
        printf("Error allocating memory on device!\n");
        exit(1);
    }
}

void randomizeParticles(particles_t *h_particles, int particles_count,
    int screen_width, int screen_height)
{
    const int max_velocity = 10;

    for (int i = 0; i < particles_count; i++)
    {
        h_particles->positions_x[i] = screen_width * (float)rand() / (float)RAND_MAX;
        h_particles->positions_y[i] = screen_height * (float)rand() / (float)RAND_MAX;

        h_particles->velocities_x[i] = 0.f;//(float)rand() / (float)RAND_MAX;
        h_particles->velocities_y[i] = 0.f;//(float)rand() / (float)RAND_MAX;

        h_particles->charges[i] = (i % 2) == 0 ? 1.f : -1.f;
        h_particles->masses[i] = 1.0f;
    }
}

void freeParticlesOnHost(particles_t *h_particles)
{
    free(h_particles->positions_x);
    free(h_particles->positions_y);

    free(h_particles->velocities_x);
    free(h_particles->velocities_y);

    free(h_particles->charges);
    free(h_particles->masses);
}

void allocateParticlesOnDevice(particles_t *d_particles, int particles_count)
{
    ssize_t size = sizeof(float) * particles_count;

    checkCudaErrors(cudaMalloc((void**)&d_particles->positions_x, size));
    checkCudaErrors(cudaMalloc((void**)&d_particles->positions_y, size));

    checkCudaErrors(cudaMalloc((void**)&d_particles->velocities_x, size));
    checkCudaErrors(cudaMalloc((void**)&d_particles->velocities_y, size));

    checkCudaErrors(cudaMalloc((void**)&d_particles->charges, size));
    checkCudaErrors(cudaMalloc((void**)&d_particles->masses, size));
}

void freeParticlesOnDevice(particles_t *d_particles)
{
    cudaFree(d_particles->positions_x);
    cudaFree(d_particles->positions_y);

    cudaFree(d_particles->velocities_x);
    cudaFree(d_particles->velocities_y);

    cudaFree(d_particles->charges);
    cudaFree(d_particles->masses);
}

void copyParticlesHtoD(particles_t *h_particles, particles_t *d_particles,
    int particles_count)
{
    ssize_t size = sizeof(float) * particles_count;

    checkCudaErrors(cudaMemcpy(d_particles->positions_x, 
        h_particles->positions_x, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_particles->positions_y,
        h_particles->positions_y, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_particles->velocities_x,
        h_particles->velocities_x, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_particles->velocities_y,
        h_particles->velocities_y, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_particles->charges,
        h_particles->charges, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_particles->masses,
        h_particles->masses, size, cudaMemcpyHostToDevice));
}

#endif // SETUP_H
