/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <unistd.h> // getopt

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>
#include <helper_math.h>

#include "calculations_utils.cuh"
#include "setup.cuh"

#define MAX(a,b) ((a > b) ? a : b)

#define REFRESH_DELAY 10 //ms

#define BLOCK_SIZE 1024

#define GPU 0
#define CPU 1

char calculations_mode = GPU;

uint particles_count = 1024 * 10;

uint window_width = 800;
uint window_height = 600;

// to pause and play the simulation
bool is_window_paused = false;

// timer to measure fps
StopWatchInterface *timer = NULL;
int fps_count = 0;
int fps_limit = 1;
float avg_fps = 0.0f;
uint g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

const char *sSDKsample = "CUDA Particle System";

// usage
void argumentsMessage();
void processArguments(int argc, char** argv);
void usage(char *name);

// declaration, forward
bool runTest(int argc, char **argv);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void timerEvent(int value);

GLubyte *h_bitmap;
GLubyte *d_bitmap;

particles_t h_particles;
particles_t d_particles;

int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    processArguments(argc, argv);
    argumentsMessage();

    runTest(argc, argv);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE); 
}

void argumentsMessage()
{
    printf("Calculations mode: %s\n", calculations_mode == GPU ? "GPU" : "CPU");
    printf("Screen resolution: %dx%d\n", window_width, window_height);
    printf("Total pixels: %d\n", window_width * window_height);
    printf("Particles count: %d\n", particles_count);
}

void processArguments(int argc, char** argv)
{
    int c;
    errno = 0;
    char* strtol_leftover;

    while((c = getopt(argc, argv, "cgp:w:h:p:")) != -1)
    {
        switch (c)
        {
        case 'c': // cpu mode
            calculations_mode = CPU;
            break;
        case 'g': // gpu mode
            calculations_mode = GPU;
            break;

        case 'w': // width
            window_width = (uint)strtol(optarg, &strtol_leftover, 10);
            if (errno != 0 || *strtol_leftover != '\0' || window_width < 1)
            {
                printf("Invalid window width\n");
                usage(argv[0]);
            }
            break;

        case 'h': // height
            window_height = (uint)strtol(optarg, &strtol_leftover, 10);
            if (errno != 0 || *strtol_leftover != '\0' || window_height < 1)
            {
                printf("Invalid window height\n");
                usage(argv[0]);
            }
            break;

        case 'p': // particles count
            particles_count = (uint)strtol(optarg, &strtol_leftover, 10);
            if (errno != 0 || *strtol_leftover != '\0' || particles_count < 1)
            {
                printf("Invalid particles count\n");
                usage(argv[0]);
            }
            break;

        case '?':
        default:
            usage(argv[0]);
        }
    }
}

void usage(char* name)
{
    fprintf(stderr, "Usage: %s [-c|-g] [-w WIDTH] [-h HEIGHT] [-p PARTICLES_COUNT]\n", name);
    fprintf(stderr, " -c CPU mode\n");
    fprintf(stderr, " -g GPU mode\n");
    exit(EXIT_FAILURE);
}

void computeFPS()
{
    fps_count++;

    if (fps_count == fps_limit)
    {
        avg_fps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fps_count = 0;
        fps_limit = (int)MAX(avg_fps, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "%s: %3.1f fps", sSDKsample, avg_fps);
    glutSetWindowTitle(fps);
}

// initialize OpenGL
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow(sSDKsample);

    // set GLUT callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutCloseFunc(cleanup);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // clear color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    // https://stackoverflow.com/questions/5877728/want-an-opengl-2d-example-vc-draw-a-rectangle
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, window_width, window_height, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    SDK_CHECK_ERROR_GL();

    return true;
}

bool runTest(int argc, char **argv)
{
    // create CUTIL timer
    sdkCreateTimer(&timer);

    h_bitmap = (GLubyte *)malloc(window_width * window_height * 3 * sizeof(GLubyte));
    if(h_bitmap == NULL)
    {
        printf("Error allocating memory for host bitmap\n");
        return false;
    }

    checkCudaErrors(cudaMalloc((void **)&d_bitmap, window_width * window_height * 3 * sizeof(GLubyte)));

    allocateParticlesOnHost(&h_particles, particles_count);
    randomizeParticles(&h_particles, particles_count, window_width, window_height);

    if(calculations_mode == GPU)
    {
        allocateParticlesOnDevice(&d_particles, particles_count);
        copyParticlesHtoD(&h_particles, &d_particles, particles_count);
        freeParticlesOnHost(&h_particles);
    }

    if (false == initGL(&argc, argv))
    {
        return false;
    }

    glutMainLoop();

    return true;
}

uint getBlocksCount(uint value)
{
    return value / BLOCK_SIZE + (value % BLOCK_SIZE ? 1 : 0);
}

void display()
{
    const float dt = 0.01f;

    sdkStartTimer(&timer);

    if(calculations_mode == CPU)
    {
        if(!is_window_paused)
        {
            h_steerParticles(h_particles, particles_count, dt, window_width, window_height);
        }

        h_colorBitmapFromParticles(h_bitmap, window_width, window_height, h_particles, particles_count);
    }
    else
    {
        if(!is_window_paused)
        {
            uint particles_blocks_count = getBlocksCount(particles_count);
            d_steerParticles<<<particles_blocks_count, BLOCK_SIZE>>>(d_particles,
                particles_count, dt, window_width, window_height);
        }

        uint pixels_blocks_count = getBlocksCount(window_width * window_height);
        d_colorBitmapFromParticles<<<pixels_blocks_count, BLOCK_SIZE>>>(d_bitmap, window_width, window_height, d_particles, particles_count);
        checkCudaErrors(cudaMemcpy(h_bitmap, d_bitmap, window_width * window_height * 3 * sizeof(GLubyte), cudaMemcpyDeviceToHost));
    }
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();
    glDrawPixels(window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, h_bitmap);
    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if(h_bitmap)
    {
        free(h_bitmap);
    }

    if(calculations_mode == GPU)
    {
        freeParticlesOnDevice(&d_particles);
    }
    else
    {
        freeParticlesOnHost(&h_particles);
    }
}

// keyboard callback
void keyboard(u_char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case 27: // ESC
            glutDestroyWindow(glutGetWindow());
            break;
        case ' ':
            is_window_paused = !is_window_paused;
            break;
        default:
            break;
    }
}
