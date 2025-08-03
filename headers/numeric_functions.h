#ifndef UNTITLED_NUMERIC_FUNCTIONS_H
#define UNTITLED_NUMERIC_FUNCTIONS_H
#include <cuda_runtime.h>

//function to sample from a wrapped cauchy distribution
__device__ float wrapped_cauchy(float mu, float sigma, curandState* state){
    float u = curand_uniform(state);
    float theta = mu + sigma * tan(M_PI * (u - 0.5f));

    //wrap the angle between -pi and pi
    if(theta > M_PI){
        theta -= 2 * M_PI;
    }
    if(theta <= -M_PI){
        theta += 2 * M_PI;
    }
    return theta;
}


// Function to compute the gradient in the X direction (partial derivative)
__device__ float gradientX(float* grid, int i, int j) {
    // Periodic boundary conditions
    int leftIndex = i - 1;
    if (leftIndex < 0) leftIndex += NN;
    int rightIndex = i + 1;
    if (rightIndex >= NN) rightIndex -= NN;
    float left = grid[leftIndex * NN + j];
    float right = grid[rightIndex * NN + j];

    return (right - left) / (2.0f * DX);  // Central difference
}

// Function to compute the gradient in the Y direction (partial derivative)
__device__ float gradientY(float* grid, int i, int j) {
    int downIndex = j - 1;
    if (downIndex < 0) downIndex += NN;
    int upIndex = j + 1;
    if (upIndex >= NN) upIndex -= NN;
    float down = grid[i * NN + downIndex];
    float up = grid[i * NN + upIndex];

    return (up - down) / (2.0f * DX);  // Central difference
}

// Function to compute the Laplacian (second derivative)
__device__ float laplacian(float* grid, int i, int j) {
    float center = grid[i * NN + j];
    int leftIndex = i - 1;
    if (leftIndex < 0) leftIndex += NN;
    int rightIndex = i + 1;
    if (rightIndex >= NN) rightIndex -= NN;
    int downIndex = j - 1;
    if (downIndex < 0) downIndex += NN;
    int upIndex = j + 1;
    if (upIndex >= NN) upIndex -= NN;
    float left = grid[leftIndex * NN + j];
    float right = grid[rightIndex * NN + j];
    float down = grid[i * NN + downIndex];
    float up = grid[i * NN + upIndex];

    float laplacian = (left + right + up + down - 4.0f * center) / (DX * DX);
    if (isnan(laplacian) || isinf(laplacian)) {
        printf("Invalid laplacian %f at (%d, %d)\n", laplacian, i, j);
        //printf("Center %f\n", center);
        //printf("Left %f\n", left);
        //printf("Right %f\n", right);
        //printf("Down %f\n", down);
        //printf("Up %f\n", up);
    }
    return laplacian;
}

// function to compute the 4th order laplacian
__device__ float fourth_order_laplacian(float* input, int i, int j){
    int im2 = (i - 2 + NN) % NN;
    int im1 = (i - 1 + NN) % NN;
    int ip1 = (i + 1) % NN;
    int ip2 = (i + 2) % NN;

    int jm2 = (j - 2 + NN) % NN;
    int jm1 = (j - 1 + NN) % NN;
    int jp1 = (j + 1) % NN;
    int jp2 = (j + 2) % NN;

    float laplacianX = (-input[im2 * NN + j] + 16 * input[im1 * NN + j] - 30 * input[i * NN + j]
                        + 16 * input[ip1 * NN + j] - input[ip2 * NN + j]) / (12 * DX * DX);

    float laplacianY = (-input[i * NN + jm2] + 16 * input[i * NN + jm1] - 30 * input[i * NN + j]
                        + 16 * input[i * NN + jp1] - input[i * NN + jp2]) /  (12 * DX * DX);

    float laplacian = laplacianX + laplacianY;
    if (i == 0 || i == NN || j == 0 || j == NN || i == 1 || i == NN - 1 || j == 1 || j == NN - 1){
        laplacian = 0;
    }

    return laplacian;
}

#endif