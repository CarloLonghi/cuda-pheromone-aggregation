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
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

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

int get_adjacency_matrix(bool* adjacency_matrix, int worm_count, float* positions, int t, float r){
    float diff_x, diff_y, dist = 0;
    for (int j = 0; j < worm_count; ++j){
        for (int k = 0; k < worm_count; ++k){
            diff_x = (positions[(t * worm_count + j) * 2] - positions[(t * worm_count + k) * 2]);
            diff_y = (positions[(t * worm_count + j) * 2 + 1] - positions[(t * worm_count + k) * 2 + 1]);
            dist = sqrt(diff_x * diff_x + diff_y * diff_y);
            if (dist <= r){
                adjacency_matrix[j*MAX_WORMS + k] = true;
            }
            else{
                adjacency_matrix[j*MAX_WORMS + k] = false;
            }
        }
    }
    return 0;
}

int reset_matrix(bool* adjacency_matrix){
    for (int i = 0; i < MAX_WORMS; ++i){
        for (int j = 0; j < MAX_WORMS; ++j){
            adjacency_matrix[i*MAX_WORMS + j] = false;
        }
    }
    return 0;
}

__global__ void initialize_rng(curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed + id, id, 0, &states[id]); // Unique seed for each thread
}

__global__ void computeCellIndices(Agent* agents, int* cellIndices, int worm_count, 
                                  float cellSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= worm_count) return;

    // Get particle position
    float2 pos = make_float2(agents[idx].x, agents[idx].y);

    // Calculate grid cell coordinates (clamp to grid bounds)
    int2 cell = getGridCell(pos);

    // Flatten 3D cell index to 1D (row-major order)
    cellIndices[idx] = flattenCellIndex(cell);
}

__global__ void buildGrid(Agent* agents, int* cellIndices, 
                        int* cellStart, int* cellEnd, int num_agents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_WORMS){

        // Special case for first particle
        if (idx == 0) {
            cellStart[cellIndices[0]] = 0;
        }
        else {
            // If this particle is in a different cell than previous
            if (cellIndices[idx] != cellIndices[idx-1]) {
                cellEnd[cellIndices[idx-1]] = idx;
                cellStart[cellIndices[idx]] = idx;
            }
        }

        // Special case for last particle
        if (idx == num_agents - 1) {
            cellEnd[cellIndices[idx]] = idx + 1;
        }
    }
}


void sortParticlesByCell(Agent* d_agents, int* d_cellIndices, int worm_count) {
    // Wrap raw pointers with Thrust device pointers
    thrust::device_ptr<Agent> particles_ptr(d_agents);
    thrust::device_ptr<int> cell_indices_ptr(d_cellIndices);

    // Sort particles by cell index (key-value sort)
    thrust::sort_by_key(cell_indices_ptr, cell_indices_ptr + worm_count, particles_ptr);
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
    float *angles = new float[MAX_WORMS * TIME];
    float align_strength, slow_factor;
    int slow_nc;

    if (argc - 1 == 13){
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
        slow_nc = std::stoi(argv[12]);
        log_worms_data = std::stoi(argv[13]);
    }
    else{
        std::cout << "The number of parameters is incorrect, it should be 11 but is " << argc - 1 << std::endl;
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
    cudaMalloc(&d_states_grids, NN * NN * sizeof(curandState));
    cudaMalloc(&potential, NN*NN*sizeof(float));
    initialize_rng<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, SEED);
    // Initialize agent positions and random states
    initAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, time(NULL), worm_count);
    //printf("Initializing agents\n");

    cudaDeviceSynchronize();
    cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    dim3 gridSize((NN + BLOCK_SIZE - 1) / BLOCK_SIZE, (NN + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    int grid_dim = GRID_DIM_X * GRID_DIM_Y;

    int* cellStart, *cellEnd, *cellIndices;
    cudaMalloc(&cellStart, grid_dim * sizeof(int));
    cudaMalloc(&cellEnd, grid_dim * sizeof(int));
    cudaMalloc(&cellIndices, worm_count * sizeof(int));

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

        cudaMemset(cellStart, 0, grid_dim * sizeof(int));
        cudaMemset(cellEnd, 0, grid_dim * sizeof(int));
        computeCellIndices<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, cellIndices, worm_count, CELL_SIZE);
        sortParticlesByCell(d_agents, cellIndices, worm_count);
        buildGrid<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, cellIndices, cellStart, cellEnd, worm_count);

        moveAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, cellStart, cellEnd, potential, /*agent_count_grid,*/ worm_count, i, sigma, align_strength, slow_factor, slow_nc);
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
            }

            // store angles
            for (int j = 0; j < worm_count; ++j) {
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
        saveAllDataToJSON(target_json, positions, h_agents ,worm_count, TIME);
    }

    // track clusters
    // float cluster_size = 0;
    // bool adjacency_matrix[MAX_WORMS][MAX_WORMS] = {false};
    // for (int i = 0; i < TIME; ++i){
    //     reset_matrix(adjacency_matrix);
    //     get_adjacency_matrix(adjacency_matrix, worm_count, positions, i, CLUSTERING_RADIUS);
    //     cluster_size += find_clusters(adjacency_matrix);
    // }
    // cluster_size /= TIME;

    // // compute worm density
    // //bool adjacency_matrix[MAX_WORMS][MAX_WORMS] = {false};
    // bool* adjacency_matrix = (bool*)malloc(MAX_WORMS * MAX_WORMS * sizeof(bool));
    // int neighbor_count = 0, nc;
    // float angle_x = 0, angle_y = 0, dir_mag = 0, dm = 0, avg_nc = 0, avg_dm = 0;
    // for (int t = 60; t < TIME; ++t){
    //     get_adjacency_matrix(adjacency_matrix, worm_count, positions, t, CLUSTERING_RADIUS);
    //     neighbor_count = 0;
    //     dir_mag = 0;
    //     for (int j = 0; j < worm_count; ++j){
    //         angle_x = 0;
    //         angle_y = 0;
    //         dm = 0;
    //         nc = 0;
    //         for (int k = 0; k < worm_count; ++k){
    //             // if (adjacency_matrix[j][k] == true && j != k){
    //             if (adjacency_matrix[j*MAX_WORMS + k] == true && j != k){
    //                 nc += 1;
    //                 angle_x += cos(angles[t * worm_count + k]);
    //                 angle_y += sin(angles[t * worm_count + k]);
    //             }
    //         }
    //         neighbor_count += nc;
    //         if (nc > 0) dm = sqrt(angle_x*angle_x + angle_y*angle_y) / nc;
    //         dir_mag += dm;
    //     }
    //     reset_matrix(adjacency_matrix);
    //     avg_nc += neighbor_count / worm_count;
    //     avg_dm += dir_mag / worm_count;
    // }
    // avg_nc /= TIME;
    // avg_dm /= TIME;

    // // float worm_density = neighbor_count / worm_count;

    // // compute distance to neighbors
    // // float avg_dist = 0, avg_dist_worm = 0, dist = 0, diffx = 0, diffy = 0;
    // // int num_neighbors;
    // // for (int i = TIME / 2; i < TIME; ++i){
    // //     reset_matrix(adjacency_matrix);
    // //     get_adjacency_matrix(adjacency_matrix, worm_count, positions, i, NEIGHBOR_RADIUS);
    // //     avg_dist_worm = 0;
    // //     for (int j = 0; j < worm_count; ++j){
    // //         dist = 0;
    // //         num_neighbors = 0;
    // //         for (int k = 0; k < worm_count; ++k){
    // //             if (j != k){
    // //                 if (adjacency_matrix[j][k] == true){
    // //                     num_neighbors += 1;
    // //                     diffx = positions[(i * worm_count + j) * 2] - positions[(i * worm_count + k) * 2];
    // //                     diffy = positions[(i * worm_count + j) * 2 + 1] - positions[(i * worm_count + k) * 2 + 1];
    // //                     dist += sqrt(diffx * diffx + diffy * diffy);
    // //                 }
    // //             }
    // //         }
    // //         if (num_neighbors > 0){
    // //             avg_dist_worm += dist / num_neighbors;
    // //         }
    // //     }
    // //     avg_dist += avg_dist_worm / worm_count;
    // // }
    // // avg_dist /= N_STEPS;  

    // // compute mean squared displacement
    // // float time_averaged_msd, mean_squared_disp, diff_x, diff_y, sq_dist = 0;
    // // for (int t = 0; t < TIME - MSD_WINDOW; t += MSD_WINDOW){
    // //     mean_squared_disp = 0;
    // //     for (int i = 0; i < worm_count; ++i){
    // //         diff_x = (positions[((t + MSD_WINDOW) * worm_count + i) * 2] - positions[(t * worm_count + i) * 2]);
    // //         if (diff_x > SPEED / DT * MSD_WINDOW){
    // //             diff_x = (positions[((t + MSD_WINDOW) * worm_count + i) * 2] - WIDTH - positions[(t * worm_count + i) * 2]);
    // //         }
    // //         if (diff_x < -SPEED / DT * MSD_WINDOW){
    // //             diff_x = (positions[((t + MSD_WINDOW) * worm_count + i) * 2] + WIDTH - positions[(t * worm_count + i) * 2]);
    // //         }            
    // //         diff_y = (positions[((t + MSD_WINDOW) * worm_count + i) * 2 + 1] - positions[(t * worm_count + i) * 2 + 1]);
    // //         if (diff_y > SPEED / DT * MSD_WINDOW){
    // //             diff_y = (positions[((t + MSD_WINDOW) * worm_count + i) * 2 + 1] - HEIGHT - positions[(t * worm_count + i) * 2 + 1]);
    // //         }
    // //         if (diff_y < -SPEED / DT * MSD_WINDOW){
    // //             diff_y = (positions[((t + MSD_WINDOW) * worm_count + i) * 2 + 1] + HEIGHT - positions[(t * worm_count + i) * 2 + 1]);
    // //         }            
    // //         sq_dist = diff_x * diff_x + diff_y * diff_y;
    // //         mean_squared_disp += (sq_dist - mean_squared_disp) / (i + 1);
    // //     }
    // //     time_averaged_msd += mean_squared_disp;
    // // }
    // // time_averaged_msd /= TIME / MSD_WINDOW;

    // // compute distance from odour spot
    // float mean_dist = 0, diff_x, diff_y;
    // for (int i = 0; i < worm_count; ++i){
    //     diff_x = positions[((TIME - 1) * worm_count + i) * 2] - MU_X;
    //     diff_y = positions[((TIME - 1) * worm_count + i) * 2 + 1] - MU_Y;
    //     mean_dist += sqrt(diff_x * diff_x + diff_y * diff_y);
    // }
    // mean_dist /= worm_count;

    // std::cout << mean_dist << " " << avg_nc << " " << avg_dm << std::endl;

    // free(adjacency_matrix);

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
