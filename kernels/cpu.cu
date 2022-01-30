#include "cpu.cuh"
#include "shared.cuh"

void h_colorBitmapFromParticles(GLubyte *bitmap, int width, int height, particles_t particles, int particles_count)
{
    for(int i = 0; i < width * height; ++i)
    {
        colorBitmapFromParticles(i, particles, NULL, particles_count, bitmap, width, height);
    }
}

void h_steerParticles(particles_t particles, int particles_count, float dt, int screen_width, int screen_height)
{
    for(int i = 0; i < particles_count; ++i)
    {
        steerParticle(i, particles, NULL, particles_count, dt, screen_width, screen_height);
    }
}
