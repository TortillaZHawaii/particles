#include "shared.cuh"
#include "../cuda_utils.cuh"

__device__ __host__ float2 calculateForce(particle_t p1, particlePosCharge_t p2)
{
    const float force_factor = 20.f;

    float2 distance = p2.position - p1.position;
    float distance_squared = dot(distance, distance);

    return -force_factor * p1.charge * p2.charge * distance /
        distance_squared * sqrt(distance_squared);
}

__device__ __host__ void keepInBounds(float2* velocity, float2* position,
    int screen_width, int screen_height)
{
    if (position->x < 0)
    {
        position->x = 0;
        velocity->x = -velocity->x;
    }
    else if (position->x >= screen_width)
    {
        position->x = screen_width;
        velocity->x = -velocity->x;
    }

    if (position->y < 0)
    {
        position->y = 0;
        velocity->y = -velocity->y;
    }
    else if (position->y >= screen_height)
    {
        position->y = screen_height;
        velocity->y = -velocity->y;
    }
}


__device__ __host__ void steerParticle(int id, particles_t particles, float* charges,
    int particles_count, float dt, int screen_width, int screen_height)
{
    if(id > particles_count)
        return;

    // if we provide charges from the shm use them, otherwise use the ones from the particles
    if(charges == NULL)
        charges = particles.charges;

    particle_t particle =
    {
        /*position*/ make_float2(particles.positions_x[id], particles.positions_y[id]),
        /*velocity*/ make_float2(particles.velocities_x[id], particles.velocities_y[id]),
        /*charge*/ charges[id],
        /*mass*/ particles.masses[id]
    };

    float2 total_force = make_float2(0.0f, 0.0f);

    // loop over other particles
    for(int i = 0; i < particles_count; i++)
    {
        if(i == id)
            continue;

        particlePosCharge_t other =
        {
            /*position*/ make_float2(particles.positions_x[i], particles.positions_y[i]),
            /*charge*/ charges[i],
        };

        float2 force = calculateForce(particle, other);

        total_force += force;
    }

    float2 acceleration = total_force / particle.mass;
    float2 velocity = particle.velocity + acceleration * dt;
    float2 position = particle.position + velocity * dt;

    keepInBounds(&velocity, &position, screen_width, screen_height);

    // update particle in soa
    particles.positions_x[id] = position.x;
    particles.positions_y[id] = position.y;

    particles.velocities_x[id] = velocity.x;
    particles.velocities_y[id] = velocity.y;
}

__device__ __host__ void setPixel(int index, GLubyte r, GLubyte g, GLubyte b, GLubyte* bitmap)
{
    index = index * 3; // rgb

    bitmap[index] = r;
    bitmap[index + 1] = g;
    bitmap[index + 2] = b;
}

__device__ __host__ void setPixel(int x, int y, int width, GLubyte r, GLubyte g, GLubyte b, 
    GLubyte* bitmap)
{
    int index = (y * width + x) * 3; // rgb

    bitmap[index] = r;
    bitmap[index + 1] = g;
    bitmap[index + 2] = b;
}

__device__ __host__ void colorBitmapFromParticles(int id, particles_t particles, float* charges,
    int particles_count, GLubyte *bitmap, int width, int height)
{
    if(id > width * height)
        return;

    if(charges == NULL)
        charges = particles.charges;
    
    int x = id % width;
    int y = id / width;

    float2 pixel_position = make_float2(x, y);

    float field_strength = 0.0f;

    for(int i = 0; i < particles_count; ++i)
    {
        float2 position = make_float2(particles.positions_x[i], particles.positions_y[i]);
        float charge = charges[i];

        float2 distance = position - pixel_position;

        float distance_squared = dot(distance, distance);

        field_strength += charge / distance_squared;
    }

    const float taint = 5.f;

    // map field strength to color
    float field_strength_mapped = field_strength * 255.0f * taint;

    // clamp to 0-255
    field_strength_mapped = clamp(abs(field_strength_mapped), 0.0f, 255.0f);

    if(field_strength > 0)
    {
        // color red (positive field strength)
        setPixel(id, (GLubyte)field_strength_mapped, 0, 0, bitmap);
    }
    else
    {
        // color blue (negative field strength)
        setPixel(id, 0, 0, (GLubyte)field_strength_mapped, bitmap);
    }
}
