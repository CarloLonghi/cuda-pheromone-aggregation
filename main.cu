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
#include "headers/gaussian_odour.h"
#include <stdbool.h>

// Function to perform DFS traversal
void dfs(bool adjMatrix[MAX_WORMS][MAX_WORMS], bool visited[MAX_WORMS], int node, int cluster[], int *size) {
    visited[node] = true;
    cluster[(*size)++] = node;
    
    for (int i = 0; i < MAX_WORMS; i++) {
        if (adjMatrix[node][i] && !visited[i]) {
            dfs(adjMatrix, visited, i, cluster, size);
        }
    }
}

// Function to find and print clusters
int find_clusters(bool adjMatrix[MAX_WORMS][MAX_WORMS]) {
    bool visited[MAX_WORMS] = {false};
    int biggest_size = 0;
    
    for (int i = 0; i < MAX_WORMS; i++) {
        if (!visited[i]) {
            int cluster[MAX_WORMS]; // Temporary storage for cluster elements
            int size = 0;
            
            dfs(adjMatrix, visited, i, cluster, &size);

            if (size > biggest_size){
                biggest_size = size;
            }
        }
    }
    return biggest_size;
}

int get_adjacency_matrix(bool adjacency_matrix[MAX_WORMS][MAX_WORMS], int worm_count, float* positions, int t, float r){
    float diff_x, diff_y, dist = 0;
    for (int j = 0; j < worm_count; ++j){
        for (int k = 0; k < worm_count; ++k){
            diff_x = (positions[(t * worm_count + j) * 2] - positions[(t * worm_count + k) * 2]);
            diff_y = (positions[(t * worm_count + j) * 2 + 1] - positions[(t * worm_count + k) * 2 + 1]);
            dist = sqrt(diff_x * diff_x + diff_y * diff_y);
            if (dist <= r){
                adjacency_matrix[j][k] = true;
            }
            else{
                adjacency_matrix[j][k] = false;
            }
        }
    }
    return 0;
}

int reset_matrix(bool adjacency_matrix[MAX_WORMS][MAX_WORMS]){
    for (int i = 0; i < MAX_WORMS; ++i){
        for (int j = 0; j < MAX_WORMS; ++j){
            adjacency_matrix[i][j] = false;
        }
    }
    return 0;
}

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
    float attractant_pheromone_strength = ATTRACTANT_PHEROMONE_STRENGTH, repulsive_pheromone_strength = REPULSIVE_PHEROMONE_STRENGTH, sigma = SIGMA, environmental_noise = ENVIRONMENTAL_NOISE;
    float attractant_pheromone_diffusion_rate = ATTRACTANT_PHEROMONE_DIFFUSION_RATE, attractant_pheromone_decay_rate = ATTRACTANT_PHEROMONE_DECAY_RATE, attractant_pheromone_secretion_rate = ATTRACTANT_PHEROMONE_SECRETION_RATE;
    float repulsive_pheromone_diffusion_rate = REPULSIVE_PHEROMONE_DIFFUSION_RATE, repulsive_pheromone_decay_rate = REPULSIVE_PHEROMONE_DECAY_RATE, repulsive_pheromone_secretion_rate = REPULSIVE_PHEROMONE_SECRETION_RATE;
    float* attractive_pheromone, * repulsive_pheromone, * h_attractive_pheromone = new float[N * N];
    float* h_repulsive_pheromone = new float[N * N], * h_potential = new float[N * N], * potential;
    int worm_count = WORM_COUNT, * agent_count_grid, * agent_count_delay;
    int * h_agent_count_grid = new int[N * N];
    //printf("Found %d arguments\n", argc-1);
    int log_worms_data = 0;

    if (argc - 1 == 9){
        attractant_pheromone_strength = std::stof(argv[1]);
        attractant_pheromone_secretion_rate = std::stof(argv[2]);
        attractant_pheromone_decay_rate = std::stof(argv[3]);        
        attractant_pheromone_diffusion_rate = std::stof(argv[4]);
        repulsive_pheromone_strength = std::stof(argv[5]);
        repulsive_pheromone_secretion_rate = std::stof(argv[6]);
        repulsive_pheromone_decay_rate = std::stof(argv[7]);
        repulsive_pheromone_diffusion_rate = std::stof(argv[8]);
        log_worms_data = std::stoi(argv[9]);
    }
    else{
        std::cout << "The number of parameters is incorrect, it should be 9 but is " << argc - 1 << std::endl;
        return 1;
    }

    const char* target_json = "/home/carlo/babots/cuda_agent_based_sim/json/agents_all_data.json";
    Agent* d_agents, *h_agents = new Agent[worm_count];
    curandState* d_states, *d_states_grids;
    bool broken = false;
    size_t size = worm_count * sizeof(Agent);
    auto* positions = new float[worm_count * TIME * 2]; // Matrix to store positions (x, y) for each agent at each timestep
    cudaMalloc(&d_agents, size);
    cudaMalloc(&d_states, worm_count * sizeof(curandState));
    cudaMalloc(&d_states_grids, N * N * sizeof(curandState));
    cudaMalloc(&potential, N*N*sizeof(float));
    initialize_rng<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, SEED);
    // Initialize agent positions and random states
    initAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, time(NULL), worm_count);
    //printf("Initializing agents\n");

    cudaDeviceSynchronize();
    cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    //initialize the agent count grid
    cudaMalloc(&agent_count_grid, N*N*sizeof(int));
    cudaMalloc(&agent_count_delay, PHEROMONE_DELAY*N*N*sizeof(int));
    initAgentDensityGrid<<<gridSize, blockSize>>>(agent_count_grid, d_agents, worm_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initAgentDensityGrid: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    //initialize the pheromone grids
    cudaMalloc(&attractive_pheromone, N*N*sizeof(float));
    cudaMalloc(&repulsive_pheromone, N*N*sizeof(float));
    //initAttractiveAndRepulsivePheromoneGrid<<<gridSize, blockSize>>>(attractive_pheromone, repulsive_pheromone, agent_count_grid9);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initAttractiveAndRepulsivePheromoneGrid: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_attractive_pheromone, attractive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_repulsive_pheromone, repulsive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //initialise the potential grid
    updatePotential<<<gridSize, blockSize>>>(potential, attractive_pheromone, repulsive_pheromone, repulsive_pheromone_strength, d_states_grids, environmental_noise);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in updatePotential: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_potential, potential, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    float mean_pheromone[TIME] = {0};

    for (int i = 0; i < N_STEPS; ++i) {
        moveAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states,  potential, /*agent_count_grid,*/ worm_count, i, sigma, attractant_pheromone_strength);
        // Check for errors in the kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        // Copy data from device to host
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
        //cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(int), cudaMemcpyDeviceToHost);

        //update all grids
        updateGrids<<<gridSize, blockSize>>>(attractive_pheromone, repulsive_pheromone, agent_count_grid, agent_count_delay, worm_count, d_agents,
        attractant_pheromone_diffusion_rate, attractant_pheromone_decay_rate, attractant_pheromone_secretion_rate,
        repulsive_pheromone_diffusion_rate, repulsive_pheromone_decay_rate, repulsive_pheromone_secretion_rate);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error in updateGrids: %s\n", cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();
        // copy data from device to host
        cudaMemcpy(h_attractive_pheromone, attractive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_repulsive_pheromone, repulsive_pheromone, N * N * sizeof(float), cudaMemcpyDeviceToHost);

        //update potential
        updatePotential<<<gridSize, blockSize>>>(potential, attractive_pheromone, repulsive_pheromone, repulsive_pheromone_strength, d_states_grids, environmental_noise);
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
            
            int t = (int)(i / LOGGING_INTERVAL);
            // Store positions
            for (int j = 0; j < worm_count; ++j) {
                positions[(t * worm_count + j) * 2] = h_agents[j].x;
                positions[(t * worm_count + j) * 2 + 1] = h_agents[j].y;
            }

            // compute pheromone density
            for (int phi = 0; phi < N; ++phi){
                for (int phj = 0; phj < N; ++phj){
                    mean_pheromone[t] += h_attractive_pheromone[phi * N + phj] + h_repulsive_pheromone[phi * N + phj];
                }
            }
            mean_pheromone[t] /= WIDTH * HEIGHT;

            if(LOG_POTENTIAL) {
                logMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/potential/potential_step_", h_potential, N, N, i);
            }
            if(LOG_AGENT_COUNT_GRID) {
                logIntMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/agent_count/agents_log_step_", h_agent_count_grid, N, N, i);
            }
            if(log_worms_data) {
                logMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/attractive_pheromone/attractive_pheromone_step_", h_attractive_pheromone, N, N, i);
                logMatrixToFile("/home/carlo/babots/cuda_agent_based_sim/logs/repulsive_pheromone/repulsive_pheromone_step_", h_repulsive_pheromone, N, N, i);
            }

        }
         

    }
    if(log_worms_data == 1) {
        saveAllDataToJSON(target_json, positions, h_agents ,worm_count, TIME);
    }

    // track clusters
    int cluster_sizes[TIME] = {0};
    bool adjacency_matrix[MAX_WORMS][MAX_WORMS] = {false};
    for (int i = 0; i < TIME; ++i){
        reset_matrix(adjacency_matrix);
        get_adjacency_matrix(adjacency_matrix, worm_count, positions, i, 1);
        cluster_sizes[i] = find_clusters(adjacency_matrix);
    }

    int biggest_size = 0;
    for (int i = 0; i < TIME; ++i){
        if (cluster_sizes[i] > biggest_size){
            biggest_size = cluster_sizes[i];
        }
    }

    // compute mean squared displacement
    float mean_squared_disp, diff_x, diff_y, sq_dist = 0;
    for (int i = 0; i < worm_count; ++i){
        diff_x = (positions[((TIME - 1) * worm_count + i) * 2] - positions[i * 2]);
        diff_y = (positions[((TIME - 1) * worm_count + i) * 2 + 1] - positions[i * 2 + 1]);
        sq_dist = diff_x * diff_x + diff_y * diff_y;
        mean_squared_disp += (sq_dist - mean_squared_disp) / (i + 1);
    }

    // compute worm density
    float worm_density = 0;
    int neighbor_count = 0;
    for (int i = 0; i < TIME; ++i){
        reset_matrix(adjacency_matrix);
        get_adjacency_matrix(adjacency_matrix, worm_count, positions, i, 10);
        neighbor_count = 0;
        for (int j = 0; j < worm_count; ++j){
            for (int k = 0; k < worm_count; ++k){
                if (adjacency_matrix[j][k] == true){
                    neighbor_count += 1;
                }
            }
        }
        worm_density += neighbor_count / worm_count;
    }
    worm_density /= TIME;

    // compute mean pheromone density
    float pheromone_density = 0;
    for (int i = 0; i < TIME; ++i){
        pheromone_density += mean_pheromone[i];
    }
    pheromone_density /= TIME;

    std::cout << biggest_size << " " << mean_squared_disp << " " << pheromone_density<< std::endl;

    cudaFree(d_agents);
    cudaFree(d_states);
    cudaFree(potential);
    cudaFree(attractive_pheromone);
    cudaFree(repulsive_pheromone);
    cudaFree(agent_count_grid);
    cudaFree(agent_count_delay);
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
