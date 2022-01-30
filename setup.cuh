#ifndef SETUP_H
#define SETUP_H

#include "particles.cuh"

void allocateParticlesOnHost(particles_t *h_particles, int particles_count);
void allocateParticlesOnDevice(particles_t *d_particles, int particles_count);

void randomizeParticles(particles_t *h_particles, int particles_count,
    int screen_width, int screen_height);

void freeParticlesOnHost(particles_t *h_particles);
void freeParticlesOnDevice(particles_t *d_particles);

void copyParticlesHtoD(particles_t *h_particles, particles_t *d_particles,
    int particles_count);

#endif // SETUP_H
