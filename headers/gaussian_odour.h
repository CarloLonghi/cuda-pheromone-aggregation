//
// Created by nema on 28/10/24.
//

#ifndef UNTITLED_GAUSSIAN_ODOUR_H
#define UNTITLED_GAUSSIAN_ODOUR_H
#include <cuda_runtime.h>

// device function to call the CUDA kernel
__device__ float computeDensityAtPoint(float x, float y, float t) {
    // Check if the point (x, y) is within the boundaries
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
        return 0.0f;  // Return 0 for points outside the boundaries
    }

    // Compute the Gaussian density value at (x, y)
    float rt = 20*60*60 + t;
    float dx = x - MU_X;
    float dy = y - MU_Y;
    float r = dx*dx + dy*dy;
    float density = pow(10, 6) * 0.2 / (4 * M_PI * 2.64 * DIFFUSION_CONSTANT * rt) * expf(-r/(4 * DIFFUSION_CONSTANT * rt));
    return density;
}
#endif //UNTITLED_GAUSSIAN_ODOUR_H