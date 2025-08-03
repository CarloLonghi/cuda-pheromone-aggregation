//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_UPDATE_MATRICES_H
#define UNTITLED_UPDATE_MATRICES_H
#include <cuda_runtime.h>
#include "numeric_functions.h"
#include "gaussian_odour.h"
#include <cmath>

//CUDA kernel to update all the grids (except the potential and the agent count grid)
__global__ void updateGrids(float* attractive_pheromone, float* repulsive_pheromone, int* agent_count_grid, 
                            int worm_count, Agent* agents, float attractant_pheromone_diffusion_rate, float attractant_pheromone_decay_rate,
                            float attractant_pheromone_secretion_rate, float repulsive_pheromone_diffusion_rate,
                            float repulsive_pheromone_decay_rate, float repulsive_pheromone_secretion_rate){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < NN && j < NN && i>=0 && j>=0) {
        // update agent_count_grid
        agent_count_grid[i * NN + j] = 0;
        for (int k = 0; k < worm_count; ++k) {
            int agent_x = (int)round(agents[k].x / DX);
            int agent_y = (int)round(agents[k].y / DY);
            if (agent_x == i && agent_y == j) {
                // printf("Agent at (%d, %d)\n", i, j);
                agent_count_grid[i * NN + j] += 1;
            }
        }

        //update attractive pheromone
        //laplacian_value = fourth_order_laplacian(attractive_pheromone, i, j);
        float new_attractive_pheromone, laplacian_attractive_pheromone = fourth_order_laplacian(attractive_pheromone, i, j);
        float new_repulsive_pheromone, laplacian_repulsive_pheromone = fourth_order_laplacian(repulsive_pheromone, i, j);
        if(agent_count_grid[i * N + j] == 0){
            new_attractive_pheromone =  attractive_pheromone[i * N + j] + DT * (attractant_pheromone_diffusion_rate * laplacian_attractive_pheromone - attractant_pheromone_decay_rate * attractive_pheromone[i * N + j]);
            new_repulsive_pheromone = repulsive_pheromone[i * N + j] + DT * (repulsive_pheromone_diffusion_rate * laplacian_repulsive_pheromone - repulsive_pheromone_decay_rate * repulsive_pheromone[i * N + j]);
        }
        else {
            new_attractive_pheromone = attractive_pheromone[i * N + j] + DT * (attractant_pheromone_diffusion_rate * laplacian_attractive_pheromone - attractant_pheromone_decay_rate * attractive_pheromone[i * N + j] + attractant_pheromone_secretion_rate * agent_count_grid[i * N + j] / (DX * DX));
            new_repulsive_pheromone = repulsive_pheromone[i * N + j] + DT * (repulsive_pheromone_diffusion_rate * laplacian_repulsive_pheromone - repulsive_pheromone_decay_rate * repulsive_pheromone[i * N + j] + repulsive_pheromone_secretion_rate * agent_count_grid[i * N + j] / (DX * DX));
        }
        if (new_attractive_pheromone < 0) new_attractive_pheromone = 0.0f;
        if (new_attractive_pheromone > MAX_CONCENTRATION) new_attractive_pheromone = MAX_CONCENTRATION;
        attractive_pheromone[i * N + j] = new_attractive_pheromone;

        if(isnan(new_attractive_pheromone) || isinf(new_attractive_pheromone)){
            printf("Invalid attractive pheromone %f at (%d, %d)\n", new_attractive_pheromone, i, j);
            printf("Laplacian value %f\n", laplacian_attractive_pheromone);
            printf("Old attractive pheromone %f\n", attractive_pheromone[i * N + j]);
        }

        //update repulsive pheromone
        //printf("Repulsive pheromone at (%d, %d): %f\n", i, j, repulsive_pheromone[i * N + j]);

        if (new_repulsive_pheromone < 0) new_repulsive_pheromone = 0.0f;
        if (new_repulsive_pheromone > MAX_CONCENTRATION) new_repulsive_pheromone = MAX_CONCENTRATION;
        repulsive_pheromone[i * N + j] = new_repulsive_pheromone;

        if(isnan(new_repulsive_pheromone) || isinf(new_repulsive_pheromone)){
            printf("Invalid repulsive pheromone %f at (%d, %d)\n", new_repulsive_pheromone, i, j);
            printf("Laplacian value %f\n", laplacian_repulsive_pheromone);
            printf("Old repulsive pheromone %f\n", repulsive_pheromone[i * N + j]);
        }
    }
}

//CUDA kernel to update the potential matrix
__global__ void updatePotential(float* potential, float* attractive_pheromone, float attractive_pheromone_strengths, float* repulsive_pheromone, float repulsive_pheromone_strength, float odour_strength, curandState* states, float environmental_noise, int timestep) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < NN && j < NN) {
        float potential_attractive_pheromone=0.0f, potential_repulsive_pheromone = 0.0f, potential_odour = 0.0f;
        //
        potential_attractive_pheromone = attractive_pheromone_strengths * attractive_pheromone[i * NN + j];// / (ATTRACTANT_PHEROMONE_SCALE + attractive_pheromone[i * NN + j]);

        potential_repulsive_pheromone = -repulsive_pheromone_strength * repulsive_pheromone[i * NN + j];// / (REPULSIVE_PHEROMONE_SCALE + repulsive_pheromone[i * NN + j]);

        potential_odour = odour_strength * computeDensityAtPoint(i * DX, j * DY, timestep);

        float noise = curand_normal(&states[i * NN + j]) * environmental_noise;
        if(noise>environmental_noise) noise = environmental_noise;
        if(noise<-environmental_noise) noise = -environmental_noise;
        potential[i * NN + j] = potential_attractive_pheromone + potential_repulsive_pheromone + potential_odour + noise;

    }
}
#endif //UNTITLED_UPDATE_MATRICES_H
