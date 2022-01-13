#ifndef PARTICLES_H
#define PARTICLES_H

struct particles_t
{
    float* positions_x;
    float* positions_y;

    float* velocities_x;
    float* velocities_y;

    float* charges;
    float* masses;
};

struct particle_t
{
    float2 position;
    float2 velocity;

    float charge;
    float mass;
};

// things required to calculate force
struct particlePosCharge_t
{
    float2 position;
    float charge;
};

#endif // PARTICLES_H
