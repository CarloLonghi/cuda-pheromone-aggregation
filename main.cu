#include <stdio.h>
#include <curand_kernel.h>
#include "include/json.hpp"
#include <fstream>
#include <iostream>
#include "headers/parameters.h"
#include "headers/init_env.h"
#include "headers/agent_update.h"
#include "headers/update_matrices.h"
#include "headers/logging.h"


int main(int argc, char* argv[]) {
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    float attractant_pheromone_strength = ATTRACTANT_PHEROMONE_STRENGTH, repulsive_pheromone_strength = REPULSIVE_PHEROMONE_STRENGTH, odor_strength = ATTRACTION_STRENGTH, sigma = SIGMA, environmental_noise = ENVIRONMENTAL_NOISE;
    float* grid, * h_grid = new float[N * N], * attractive_pheromone, * repulsive_pheromone, * h_attractive_pheromone = new float[N * N];
    float* h_repulsive_pheromone = new float[N * N], * h_potential = new float[N * N], * potential;
    int worm_count = WORM_COUNT, exp_number = 0, * agent_count_grid, * h_agent_count_grid = new int[N * N];;


    printf("Found %d arguments\n", argc-1);
    switch (argc-1) {
        case 2:
            if(std::isdigit(argv[1][0]) && std::isdigit(argv[2][0])){
                /*attractant_pheromone_strength = std::stof(argv[1]);
                printf("Attractant pheromone strength: %.10f\n", attractant_pheromone_strength);
                repulsive_pheromone_strength = std::stof(argv[2]);
                printf("Repulsive pheromone strength: %.10f\n", repulsive_pheromone_strength);*/
                sigma = std::stof(argv[1]);
                printf("Sigma: %.10f\n", sigma);
                environmental_noise = std::stof(argv[2]);
                printf("Environmental noise: %.10f\n", environmental_noise);
            }
            break;

        case 5:
            if(std::isdigit(argv[1][0]) && std::isdigit(argv[2][0]) && std::isdigit(argv[3][0]) && std::isdigit(argv[4][0])  && std::isdigit(argv[5][0])){
                exp_number = std::stoi(argv[1]);
                printf("Experiment number: %d\n", exp_number);
                worm_count = std::stoi(argv[2]);
                printf("Worm count: %d\n", worm_count);
                attractant_pheromone_strength = std::stof(argv[3]);
                printf("Attractant pheromone strength: %.10f\n", attractant_pheromone_strength);
                repulsive_pheromone_strength = -std::stof(argv[4]);
                printf("Repulsive pheromone strength: %.10f\n", repulsive_pheromone_strength);
                int using_odor = std::stoi(argv[5]);
                if(using_odor == 0){
                    odor_strength = 0.0f;
                }
                printf("Odor strength: %f\n", odor_strength);
            }
            break;
        case 0:
            printf("No input arguments provided.\n");
            break;
    }
    Agent* d_agents, *h_agents = new Agent[worm_count];
    curandState* d_states, *d_states_grids;
    bool broken = false;
    size_t size = worm_count * sizeof(Agent);

    auto* positions = new float[worm_count * N_STEPS * 2]; // Matrix to store positions (x, y) for each agent at each timestep
    auto* angles = new float[worm_count * N_STEPS]; // Matrix to store angles for each agent at each timestep
    auto* velocities = new float[worm_count * N_STEPS]; // Matrix to store velocities for each agent at each timestep
    cudaMalloc(&d_agents, size);
    cudaMalloc(&d_states, worm_count * sizeof(curandState));
    cudaMalloc(&d_states_grids, N * N * sizeof(curandState));
    cudaMalloc(&grid, N*N*sizeof(float));
    cudaMalloc(&potential, N*N*sizeof(float));

    // Initialize agent positions and random states
    initAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, time(NULL), worm_count);
    printf("Initializing agents\n");

    cudaDeviceSynchronize();
    cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    //initialize the agent count grid
    cudaMalloc(&agent_count_grid, N*N*sizeof(int));
    initAgentDensityGrid<<<gridSize, blockSize>>>(agent_count_grid, d_agents, worm_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initAgentDensityGrid: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

// Initialize the chemical grid concentration
    initGrid<<<gridSize, blockSize>>>(grid, d_states_grids);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initGrid: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_grid, grid, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //initialize the pheromone grids
    cudaMalloc(&attractive_pheromone, N*N*sizeof(float));
    cudaMalloc(&repulsive_pheromone, N*N*sizeof(float));
    initAttractiveAndRepulsivePheromoneGrid<<<gridSize, blockSize>>>(attractive_pheromone, repulsive_pheromone, agent_count_grid);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initAttractiveAndRepulsivePheromoneGrid: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_attractive_pheromone, attractive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_repulsive_pheromone, repulsive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //initialise the potential grid
    updatePotential<<<gridSize, blockSize>>>(potential, grid, attractive_pheromone, repulsive_pheromone, attractant_pheromone_strength, repulsive_pheromone_strength, odor_strength, d_states_grids, environmental_noise);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in updatePotential: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_potential, potential, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Move agents in a loop
    for (int i = 0; i < N_STEPS; ++i) {
        //printf("Step %d\n", i);

        //copy the agent count grid to the device
        cudaMemcpy(agent_count_grid, h_agent_count_grid, N * N * sizeof(int), cudaMemcpyHostToDevice);
        moveAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, potential, agent_count_grid, worm_count, i, sigma);
        // Check for errors in the kernel launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        // Copy data from device to host
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        // Store positions, velocities and angles in the matrices
        for (int j = 0; j < worm_count; ++j) {
            positions[(i * worm_count + j) * 2] = h_agents[j].x;
            positions[(i * worm_count + j) * 2 + 1] = h_agents[j].y;

            angles[i * worm_count + j] = h_agents[j].angle;

            velocities[i * worm_count + j] = h_agents[j].speed;
        }

        //copy the repulsive pheromone grid to the device
        cudaMemcpy(attractive_pheromone, h_attractive_pheromone, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(repulsive_pheromone, h_repulsive_pheromone, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grid, h_grid, N * N * sizeof(float), cudaMemcpyHostToDevice);

        //update all grids
        updateGrids<<<gridSize, blockSize>>>(grid, attractive_pheromone, repulsive_pheromone, agent_count_grid);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in updateGrids: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        // copy data from device to host
        cudaMemcpy(h_grid, grid, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_attractive_pheromone, attractive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_repulsive_pheromone, repulsive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);

        //update potential
        updatePotential<<<gridSize, blockSize>>>(potential, grid, attractive_pheromone, repulsive_pheromone, attractant_pheromone_strength, repulsive_pheromone_strength, odor_strength, d_states_grids, environmental_noise);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in updatePotential: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        cudaMemcpy(h_potential, potential, N * N * sizeof(float), cudaMemcpyDeviceToHost);

        //check if any value in grid is invalid
        if (DEBUG){
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (isnan(h_grid[i * N + j]) || isinf(h_grid[i * N + j])) {
                        printf("Invalid concentration %f at (%d, %d)\n", h_grid[i * N + j], i, j);
                        broken = true;
                        break;
                    }
                    if (isnan(h_attractive_pheromone[i * N + j]) || isinf(h_attractive_pheromone[i * N + j])) {
                        printf("Invalid attractive pheromone %f at (%d, %d)\n", h_attractive_pheromone[i * N + j], i, j);
                        broken = true;
                        break;
                    }
                    if (isnan(h_repulsive_pheromone[i * N + j]) || isinf(h_repulsive_pheromone[i * N + j])) {
                        printf("Invalid repulsive pheromone %f at (%d, %d)\n", h_repulsive_pheromone[i * N + j], i, j);
                        broken = true;
                        break;
                    }
                    if (isnan(h_potential[i * N + j]) || isinf(h_potential[i * N + j])) {
                        printf("Invalid potential %f at (%d, %d)\n", h_potential[i * N + j], i, j);
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
            if(LOG_POTENTIAL) {
                logMatrixToFile("/home/nema/CLionProjects/untitled/logs/potential/potential_step_", h_potential, N, N, i);
            }
            if(LOG_AGENT_COUNT_GRID) {
                logIntMatrixToFile("/home/nema/CLionProjects/untitled/logs/agent_count/agents_log_step_", h_agent_count_grid, N, N, i);
            }
            if(LOG_GRID) {
                logMatrixToFile("/home/nema/CLionProjects/untitled/logs/chemical_concentration/chemical_concentration_step_", h_grid, N, N, i);
            }
            if(LOG_PHEROMONES) {
                logMatrixToFile("/home/nema/CLionProjects/untitled/logs/attractive_pheromone/attractive_pheromone_step_", h_attractive_pheromone, N, N, i);
                logMatrixToFile("/home/nema/CLionProjects/untitled/logs/repulsive_pheromone/repulsive_pheromone_step_", h_repulsive_pheromone, N, N, i);
            }

        }

    }
    if(LOG_GENERIC_TARGET_DATA) {
        saveAllDataToJSON("/home/nema/CLionProjects/untitled/agents_all_data.json", positions, velocities, angles, h_agents ,worm_count, N_STEPS);
    }

    /*if(LOG_TRAJECTORIES) {
        savePositionsToJSON("/home/nema/CLionProjects/untitled/agents_log.json", positions, worm_count, N_STEPS);
    }
    if(LOG_VELOCITIES) {
        savePositionsToJSON("/home/nema/CLionProjects/untitled/agents_velocities_log.json", velocities, worm_count, N_STEPS, true);
    }
    if(LOG_ANGLES) {
        savePositionsToJSON("/home/nema/CLionProjects/untitled/agents_angles_log.json", angles, worm_count, N_STEPS, true);
    } //
    saveInsideAreaToJSON("/home/nema/CLionProjects/untitled/inside_area.json", h_agents, worm_count, N_STEPS);*/
    cudaFree(d_agents);
    cudaFree(d_states);
    cudaFree(grid);
    cudaFree(potential);
    cudaFree(attractive_pheromone);
    cudaFree(repulsive_pheromone);
    cudaFree(agent_count_grid);
    delete[] h_agents;
    delete[] h_grid;
    delete[] h_potential;
    delete[] h_attractive_pheromone;
    delete[] h_repulsive_pheromone;
    delete[] h_agent_count_grid;
    delete[] positions;
    delete[] angles;
    delete[] velocities;


    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize the events
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the elapsed time
    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}
