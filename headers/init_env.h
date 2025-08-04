//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_INIT_ENV_H
#define UNTITLED_INIT_ENV_H
#include <cuda_runtime.h>
#include <random>
#include "../include/json.hpp"
#include <cmath>
using json = nlohmann::json;


struct Agent {
    float x, y, angle, speed, previous_potential, cumulative_potential;  // Position in 2D space
};

// CUDA kernel to initialize the position of each agent
__global__ void initAgents(Agent* agents, curandState* states, unsigned long seed, int worm_count) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {
        curand_init(seed, id, 0, &states[id]);
        if (ENABLE_RANDOM_INITIAL_POSITIONS) {
            agents[id].x = curand_uniform(&states[id]) * WIDTH;
            agents[id].y = curand_uniform(&states[id]) * HEIGHT;
        } else {
            //initialise in a random position at the center of the environment
            float angle = curand_uniform(&states[id]) * 2 * M_PI;
            agents[id].x = WIDTH / 2 + cos(angle) * curand_uniform(&states[id]) * INITIAL_AREA_SIZE;
            agents[id].y = HEIGHT / 2 + sin(angle) * curand_uniform(&states[id]) * INITIAL_AREA_SIZE;           
        }
        //generate angle in the range [-pi, pi]
        agents[id].angle =(2.0f * curand_uniform(&states[id]) - 1.0f) * M_PI;
        agents[id].speed = SPEED;
        agents[id].previous_potential = 0.0f;
        agents[id].cumulative_potential = 0.0f;
    }
}


// CUDA kernel to initialize the pheromone grids
__global__ void initAttractiveAndRepulsivePheromoneGrid(float* attractive_pheromone, float attractive_pheromone_secretion_rate, float attractive_pheromone_decay_rate, 
                                                    float* repulsive_pheromone, float repulsive_pheromone_secretion_rate, float repulsive_pheromone_decay_rate,
                                                    int* agent_density_grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < NN && j < NN) {
        if(agent_density_grid[i * NN + j] == 0){
            attractive_pheromone[i * NN + j] = 0.0f;
            repulsive_pheromone[i * NN + j] = 0.0f;
        }
        else {
            attractive_pheromone[i * NN + j] = attractive_pheromone_secretion_rate * attractive_pheromone_decay_rate *
                                              (float) agent_density_grid[i * NN + j] / (DX * DX);
            repulsive_pheromone[i * NN + j] = repulsive_pheromone_secretion_rate * repulsive_pheromone_decay_rate *
                                             (float) agent_density_grid[i * NN + j] / (DX * DX);
        }
        //printf("Repulsive pheromone at (%d, %d): %f\n", i, j, repulsive_pheromone[i * NN + j]);
    }
}

//CUDA kernel to initialise the agent count grid
__global__ void initAgentDensityGrid(int* agent_count_grid, Agent* agents, int worm_count){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < NN && j < NN) {
        agent_count_grid[i * NN + j] = 0;
        for (int k = 0; k < worm_count; ++k) {
            int agent_x = (int)round(agents[k].x / DX);
            int agent_y = (int)round(agents[k].y / DY);
            if (agent_x == i && agent_y == j) {
                // printf("Agent at (%d, %d)\n", i, j);
                agent_count_grid[i * NN + j] += 1;
            }
        }
    }
}
#endif //UNTITLED_INIT_ENV_H
