//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_UPDATE_MATRICES_H
#define UNTITLED_UPDATE_MATRICES_H
#include <cuda_runtime.h>
#include "numeric_functions.h"
#include <cmath>

//CUDA kernel to update all the grids (except the potential and the agent count grid)
__global__ void updateGrids(float* attractive_pheromone, float* new_attractive_pheromone, float* repulsive_pheromone, 
                            float* new_repulsive_pheromone, int* agent_count_grid, int* agent_count_delay, 
                            int worm_count, Agent* agents, float attractant_pheromone_diffusion_rate, float attractant_pheromone_decay_rate,
                            float attractant_pheromone_secretion_rate, float repulsive_pheromone_diffusion_rate,
                            float repulsive_pheromone_decay_rate, float repulsive_pheromone_secretion_rate){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N && i>=0 && j>=0) {
        // update agent_count_grid
        // agent_count_grid[i * N + j] = 0;
        // for (int k = 0; k < worm_count; ++k) {
        //     int agent_x = (int)round(agents[k].x / DX);
        //     int agent_y = (int)round(agents[k].y / DY);
        //     if (agent_x == i && agent_y == j) {
        //         // printf("Agent at (%d, %d)\n", i, j);
        //         agent_count_grid[i * N + j] += 1;
        //     }
        // }

        //update attractive pheromone
        //laplacian_value = fourth_order_laplacian(attractive_pheromone, i, j);
        float laplacian_attractive_pheromone = fourth_order_laplacian(attractive_pheromone, i, j);
        float laplacian_repulsive_pheromone = fourth_order_laplacian(repulsive_pheromone, i, j);
        new_attractive_pheromone[i * N + j] = attractive_pheromone[i * N + j] + DT * (attractant_pheromone_diffusion_rate * laplacian_attractive_pheromone - attractant_pheromone_decay_rate * attractive_pheromone[i * N + j] + attractant_pheromone_secretion_rate * agent_count_delay[(PHEROMONE_DELAY - 1) * N*N + i * N + j] / (DX * DY));
        new_repulsive_pheromone[i * N + j] = repulsive_pheromone[i * N + j] + DT * (repulsive_pheromone_diffusion_rate * laplacian_repulsive_pheromone - repulsive_pheromone_decay_rate * repulsive_pheromone[i * N + j] + repulsive_pheromone_secretion_rate * agent_count_delay[(PHEROMONE_DELAY - 1) * N*N + i * N + j] / (DX * DY));                
        if (new_attractive_pheromone[i * N + j] < 0) new_attractive_pheromone[i * N + j] = 0.0f;
        if (new_attractive_pheromone[i * N + j] > MAX_CONCENTRATION) new_attractive_pheromone[i * N + j] = MAX_CONCENTRATION;

        if(isnan(new_attractive_pheromone[i * N + j]) || isinf(new_attractive_pheromone[i * N + j])){
            printf("Invalid attractive pheromone %f at (%d, %d)\n", new_attractive_pheromone[i * N + j], i, j);
            printf("Laplacian value %f\n", laplacian_attractive_pheromone);
            printf("Old attractive pheromone %f\n", attractive_pheromone[i * N + j]);
        }

        if (new_repulsive_pheromone[i * N + j] < 0) new_repulsive_pheromone[i * N + j] = 0.0f;
        if (new_repulsive_pheromone[i * N + j] > MAX_CONCENTRATION) new_repulsive_pheromone[i * N + j] = MAX_CONCENTRATION;

        if(isnan(new_repulsive_pheromone[i * N + j]) || isinf(new_repulsive_pheromone[i * N + j])){
            printf("Invalid repulsive pheromone %f at (%d, %d)\n", new_repulsive_pheromone[i * N + j], i, j);
            printf("Laplacian value %f\n", laplacian_repulsive_pheromone);
            printf("Old repulsive pheromone %f\n", repulsive_pheromone[i * N + j]);
        }
        // update agent_count_delay
        for (int t = PHEROMONE_DELAY - 1; t > 0; --t){
            agent_count_delay[t * N*N + i * N + j] = agent_count_delay[(t - 1) * N*N + i * N + j];
        }
        agent_count_delay[i * N + j] =  agent_count_grid[i * N + j];
    }
}

//CUDA kernel to update the grid of the chemical concentration using a reaction-diffusion equation
__global__ void updateGrid(float* grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {

        float laplacian_value = laplacian(grid, i, j);

        float new_concentration = grid[i * N + j] + DT * (DIFFUSION_CONSTANT * laplacian_value - GAMMA * grid[i * N + j]);
        if (new_concentration < 0) new_concentration = 0.0f;
        if (new_concentration > MAX_CONCENTRATION) new_concentration = MAX_CONCENTRATION;
        //check if the grid is a valid float number
        if (isnan(new_concentration) || isinf(new_concentration)) {
            printf("Invalid concentration %f at (%d, %d)\n", new_concentration, i, j);
            printf("Laplacian value %f\n", laplacian_value);
            printf("Old concentration %f\n", grid[i * N + j]);

        }

        grid[i * N + j] = new_concentration;
    }
}

//CUDA kernel to update the potential matrix
__global__ void updatePotential(float* potential, float* attractive_pheromone, float* repulsive_pheromone, float repulsive_pheromone_strength, curandState* states, float environmental_noise) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        float potential_attractive_pheromone=0.0f, potential_repulsive_pheromone = 0.0f;
        //
        potential_attractive_pheromone = attractive_pheromone[i * N + j];// / (ATTRACTANT_PHEROMONE_SCALE + attractive_pheromone[i * N + j]);

        potential_repulsive_pheromone = -repulsive_pheromone_strength * repulsive_pheromone[i * N + j];// / (REPULSIVE_PHEROMONE_SCALE + repulsive_pheromone[i * N + j]);


        float noise = curand_normal(&states[i * N + j]) * environmental_noise;
        if(noise>environmental_noise) noise = environmental_noise;
        if(noise<-environmental_noise) noise = -environmental_noise;
        potential[i * N + j] = potential_attractive_pheromone + potential_repulsive_pheromone + noise;

    }
}
#endif //UNTITLED_UPDATE_MATRICES_H
