#include <stdio.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include "headers/parameters.h"
#include "headers/init_env.h"
#include "headers/agent_update.h"
#include "headers/update_matrices.h"
#include "headers/logging.h"
#include "headers/gaussian_odour.h"
#include <stdbool.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>

__global__ void initialize_rng(curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed + id, id, 0, &states[id]); // Unique seed for each thread
}


int main(int argc, char* argv[]) {
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    float attractant_pheromone_strength, repulsive_pheromone_strength, sigma = SIGMA, environmental_noise = ENVIRONMENTAL_NOISE;
    float attractant_pheromone_diffusion_rate, attractant_pheromone_decay_rate, attractant_pheromone_secretion_rate;
    float repulsive_pheromone_diffusion_rate, repulsive_pheromone_decay_rate, repulsive_pheromone_secretion_rate;
    float odour_strength;
    float* attractive_pheromone, * repulsive_pheromone, * h_attractive_pheromone = new float[NN * NN];
    float* h_repulsive_pheromone = new float[NN * NN], * h_potential = new float[NN * NN], * potential;
    int worm_count = WORM_COUNT, * agent_count_grid;
    int * h_agent_count_grid = new int[NN * NN];
    int log_worms_data = 0;
    float align_strength, slow_factor, attr_strength, rep_strength;
    char* target_json;
    char *exp_num;

    if (argc - 1 == 16){
        attractant_pheromone_strength = std::stof(argv[1]);
        attractant_pheromone_secretion_rate = std::stof(argv[2]);
        attractant_pheromone_decay_rate = std::stof(argv[3]);        
        attractant_pheromone_diffusion_rate = std::stof(argv[4]);
        repulsive_pheromone_strength = std::stof(argv[5]);
        repulsive_pheromone_secretion_rate = std::stof(argv[6]);
        repulsive_pheromone_decay_rate = std::stof(argv[7]);
        repulsive_pheromone_diffusion_rate = std::stof(argv[8]);
        odour_strength = std::stof(argv[9]);
        align_strength = std::stof(argv[10]);
        slow_factor = std::stof(argv[11]);
        attr_strength = std::stof(argv[12]);
        rep_strength = std::stof(argv[13]);
        log_worms_data = std::stoi(argv[14]);
        target_json = argv[15];
        exp_num = argv[16];
    }
    else{
        std::cout << "The number of parameters is incorrect, it should be 16 but is " << argc - 1 << std::endl;
        return 1;
    }

    std::random_device rd;

    Agent* d_agents, *h_agents = new Agent[worm_count];
    curandState* d_states, *d_states_grids;
    bool broken = false;
    size_t size = worm_count * sizeof(Agent);
    auto* positions = new float[worm_count * (N_STEPS / LOGGING_INTERVAL) * 2]; // Matrix to store positions (x, y) for each agent at each timestep
    float *angles = new float[worm_count * (N_STEPS / LOGGING_INTERVAL)];
    cudaMalloc(&d_agents, size);
    cudaMalloc(&d_states, worm_count * sizeof(curandState));
    cudaMalloc(&d_states_grids, NN * NN * sizeof(curandState));
    cudaMalloc(&potential, NN*NN*sizeof(float));
    initialize_rng<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, rd());
    // Initialize agent positions and random states
    initAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, rd(), worm_count);
    //printf("Initializing agents\n");

    cudaDeviceSynchronize();
    cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dim3 gridSize((NN + BLOCK_SIZE - 1) / BLOCK_SIZE, (NN + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    //initialize the agent count grid
    cudaMalloc(&agent_count_grid, NN*NN*sizeof(int));
    initAgentDensityGrid<<<gridSize, blockSize>>>(agent_count_grid, d_agents, worm_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initAgentDensityGrid: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_agent_count_grid, agent_count_grid, NN * NN * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    //initialize the pheromone grids
    cudaMalloc(&attractive_pheromone, NN*NN*sizeof(float));
    cudaMalloc(&repulsive_pheromone, NN*NN*sizeof(float));
    //initAttractiveAndRepulsivePheromoneGrid<<<gridSize, blockSize>>>(attractive_pheromone, repulsive_pheromone, agent_count_grid9);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initAttractiveAndRepulsivePheromoneGrid: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_attractive_pheromone, attractive_pheromone, NN * NN * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_repulsive_pheromone, repulsive_pheromone, NN * NN * sizeof(float), cudaMemcpyDeviceToHost);

    //initialise the potential grid
    updatePotential<<<gridSize, blockSize>>>(potential, attractive_pheromone, attractant_pheromone_strength, repulsive_pheromone, repulsive_pheromone_strength, odour_strength, d_states_grids, environmental_noise, 0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in updatePotential: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_potential, potential, NN * NN * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_STEPS; ++i) {

        //update all grids
        updateGrids<<<gridSize, blockSize>>>(attractive_pheromone, repulsive_pheromone, agent_count_grid, worm_count, d_agents,
        attractant_pheromone_diffusion_rate, attractant_pheromone_decay_rate, attractant_pheromone_secretion_rate,
        repulsive_pheromone_diffusion_rate, repulsive_pheromone_decay_rate, repulsive_pheromone_secretion_rate);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in updateGrids: %s\n", cudaGetErrorString(err));
        }

        moveAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, potential, /*agent_count_grid,*/ worm_count, i, sigma, align_strength, slow_factor, attr_strength, rep_strength);
        // Check for errors in the kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        // Copy data from device to host
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        //update potential
        updatePotential<<<gridSize, blockSize>>>(potential, attractive_pheromone, attractant_pheromone_strength, repulsive_pheromone, repulsive_pheromone_strength, odour_strength, d_states_grids, environmental_noise, i);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in updatePotential: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        cudaMemcpy(h_potential, potential, NN * NN * sizeof(float), cudaMemcpyDeviceToHost);        

        cudaDeviceSynchronize();
        // copy data from device to host
        cudaMemcpy(h_attractive_pheromone, attractive_pheromone, NN * NN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_repulsive_pheromone, repulsive_pheromone, NN * NN * sizeof(float), cudaMemcpyDeviceToHost);

        //check if any value in grid is invalid
        if (DEBUG){
            for (int i = 0; i < NN; ++i) {
                for (int j = 0; j < NN; ++j) {
                    if (isnan(h_attractive_pheromone[i * NN + j]) || isinf(h_attractive_pheromone[i * NN + j])) {
                        printf("Invalid attractive pheromone %f at (%d, %d)\n", h_attractive_pheromone[i * NN + j], i, j);
                        broken = true;
                        break;
                    }
                    if (isnan(h_repulsive_pheromone[i * NN + j]) || isinf(h_repulsive_pheromone[i * NN + j])) {
                        printf("Invalid repulsive pheromone %f at (%d, %d)\n", h_repulsive_pheromone[i * NN + j], i, j);
                        broken = true;
                        break;
                    }
                    if (isnan(h_potential[i * NN + j]) || isinf(h_potential[i * NN + j])) {
                        printf("Invalid potential %f at (%d, %d)\n", h_potential[i * NN + j], i, j);
                        broken = true;
                        break;
                    }
                }
            }
        }
        if (broken) {
            break;
        }
        // Save positions to JSON every LOGGING_INTERVAL steps
        if (i % LOGGING_INTERVAL == 0) {

            int t = (int)(i / LOGGING_INTERVAL);
            // Store positions
            for (int j = 0; j < worm_count; ++j) {
                positions[(t * worm_count + j) * 2] = h_agents[j].x;
                positions[(t * worm_count + j) * 2 + 1] = h_agents[j].y;
                angles[t * worm_count + j] = h_agents[j].angle;
            }          

            if(LOG_POTENTIAL) {
                logMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/potential/potential_step_", h_potential, NN, NN, i);
            }
            if(LOG_AGENT_COUNT_GRID) {
                logIntMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/agent_count/agents_log_step_", h_agent_count_grid, NN, NN, i);
            }
            if(log_worms_data) {
                logMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/attractive_pheromone/attractive_pheromone_step_", h_attractive_pheromone, NN, NN, (int) i/LOGGING_INTERVAL);
                logMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/repulsive_pheromone/repulsive_pheromone_step_", h_repulsive_pheromone, NN, NN, (int) i/LOGGING_INTERVAL);
            }

        }
         

    }
    if(log_worms_data == 1) {
        saveAllDataToJSON(target_json, exp_num, positions, angles, h_agents ,worm_count, N_STEPS / LOGGING_INTERVAL);
    }

    std::cout << 0 <<std::endl;

    cudaFree(d_agents);
    cudaFree(d_states);
    cudaFree(potential);
    cudaFree(attractive_pheromone);
    cudaFree(repulsive_pheromone);
    cudaFree(agent_count_grid);
    delete[] h_agents;
    delete[] h_potential;
    delete[] h_attractive_pheromone;
    delete[] h_repulsive_pheromone;
    delete[] h_agent_count_grid;
    delete[] positions;

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize the events
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the elapsed time
    //std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}
