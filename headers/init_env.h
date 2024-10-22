//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_INIT_ENV_H
#define UNTITLED_INIT_ENV_H
#include <cuda_runtime.h>


struct Agent {
    float x, y, angle, speed, previous_potential, cumulative_potential;  // Position in 2D space
    int state;  // State of the agent: -1 stopped, 0 moving, 1 pirouette
    int is_agent_in_target_area;
    int first_timestep_in_target_area, steps_in_target_area;
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
            //initialise in a random position inside the square centered at WIDTH/4, HEIGHT/4 with side length DX*INITIAL_AREA_NUMBER_OF_CELLS
           agents[id].x = WIDTH / 4 - INITIAL_AREA_NUMBER_OF_CELLS/2 * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
           agents[id].y = HEIGHT / 2 - INITIAL_AREA_NUMBER_OF_CELLS/2  * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
        }
        //generate angle in the range [-pi, pi]
        agents[id].angle =(2.0f * curand_uniform(&states[id]) - 1.0f) * M_PI;
        agents[id].speed = SPEED;
        agents[id].state = 0;
        agents[id].previous_potential = 0.0f;
        agents[id].cumulative_potential = 0.0f;
        agents[id].is_agent_in_target_area = 0;
        agents[id].first_timestep_in_target_area = -1;
        agents[id].steps_in_target_area = 0;
    }
}

// CUDA kernel to initialize the chemical grid concentration
__global__ void initGrid(float* grid, curandState* states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        //place 100 units of chemical in the square in the middle of the grid with length 20
        if (i >= 3 * N / 4 - TARGET_AREA_SIDE_LENGTH/2 && i < 3 * N / 4 + TARGET_AREA_SIDE_LENGTH/2 && j >= N / 2 - TARGET_AREA_SIDE_LENGTH/2 && j < N / 2 + TARGET_AREA_SIDE_LENGTH/2) {
        //if (i >=N / 2 - TARGET_AREA_SIDE_LENGTH/2 && i <  N / 2 + TARGET_AREA_SIDE_LENGTH/2 && j >= N / 2 - TARGET_AREA_SIDE_LENGTH/2 && j < N / 2 + TARGET_AREA_SIDE_LENGTH/2) {
            grid[i * N + j] = MAX_CONCENTRATION * (1.0f + curand_normal(&states[i*N+j]));
        } else{
            grid[i * N + j] = 0.0f;
        }
    }
}


// CUDA kernel to initialize the chemical grid with two squares of chemical placed in the lower left and upper right corners. size 10x10 cells each
__global__ void initGridWithTwoSquares(float* grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        int upper_right_center_x = 3*N/4;
        int upper_right_center_y = 3*N/4;
        int lower_left_center_x = N/4;
        int lower_left_center_y = N/4;
        if ((i >= upper_right_center_x - 5 && i < upper_right_center_x + 5 && j >= upper_right_center_y - 5 && j < upper_right_center_y + 5) || (i >= lower_left_center_x - 5 && i < lower_left_center_x + 5 && j >= lower_left_center_y - 5 && j < lower_left_center_y + 5)) {
            grid[i * N + j] = MAX_CONCENTRATION;
        } else{
            grid[i * N + j] = 0.0f;
        }
    }
}


// CUDA kernel to initialize the pheromone grids
__global__ void initAttractiveAndRepulsivePheromoneGrid(float* attractive_pheromone, float* repulsive_pheromone, int* agent_density_grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        if(agent_density_grid[i * N + j] == 0){
            attractive_pheromone[i * N + j] = 0.0f;
            repulsive_pheromone[i * N + j] = 0.0f;
        }
        else {
            attractive_pheromone[i * N + j] = ATTRACTANT_PHEROMONE_SECRETION_RATE * ATTRACTANT_PHEROMONE_DECAY_RATE *
                                              (float) agent_density_grid[i * N + j] / (DX * DX);
            repulsive_pheromone[i * N + j] = REPULSIVE_PHEROMONE_SECRETION_RATE * REPULSIVE_PHEROMONE_DECAY_RATE *
                                             (float) agent_density_grid[i * N + j] / (DX * DX);
        }
        //printf("Repulsive pheromone at (%d, %d): %f\n", i, j, repulsive_pheromone[i * N + j]);
    }
}

//CUDA kernel to initialise the agent count grid
__global__ void initAgentDensityGrid(int* agent_count_grid, Agent* agents, int worm_count){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        agent_count_grid[i * N + j] = 0;
        for (int k = 0; k < worm_count; ++k) {
            int agent_x = (int)(agents[k].x / DX);
            int agent_y = (int)(agents[k].y / DX);
            if (agent_x == i && agent_y == j) {
                // printf("Agent at (%d, %d)\n", i, j);
                agent_count_grid[i * N + j] += 1;
            }
        }
    }
}
#endif //UNTITLED_INIT_ENV_H
