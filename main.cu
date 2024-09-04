#include <stdio.h>
#include <curand_kernel.h>
#include "include/json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

#define N 128                        // Grid size
#define WORM_COUNT 500                 // Number of agents
#define WIDTH 20.0f          // Width of the 2D space
#define HEIGHT 20.0f         // Height of the 2D space
#define BLOCK_SIZE 32        // CUDA block size
#define N_STEPS 5000         // Number of simulation steps
#define LOGGING_INTERVAL 10    // Logging interval for saving positions
#define SPEED 0.015f            // Constant speed at which agents move
//#define DX WIDTH/N               // Grid spacing
#define LAMBDA 0.9f             //persistance of the movement
#define DRIFT_FACTOR 0.01f       //drift factor
#define SENSING_RANGE 1        //sensing range of the agents
#define MAX_CONCENTRATION 100.0f //maximum concentration of the chemical
#define DT 1.0f                //time step
#define GAMMA 0.0001f             //decay rate of the chemical
#define DIFFUSION_CONSTANT 0.001f                  //diffusion rate of the chemical
#define ATTRACTION_STRENGTH 0.111f
#define ATTRACTION_SCALE 1.5f
#define ODOR_THRESHOLD 0.1f
#define DEBUG false
#define SIGMA 0.001f
#define INITIAL_AREA_NUMBER_OF_CELLS 10 //defines the side length of the square where the agents are initialized in terms of number of cells
//pheromone parameters
#define ATTRACTANT_PHEROMONE_SCALE 1.5f
#define ATTRACTANT_PHEROMONE_STRENGTH 0.00411f
#define ATTRACTANT_PHEROMONE_DECAY_RATE 0.001f
#define ATTRACTANT_PHEROMONE_SECRETION_RATE 0.001f
#define ATTRACTANT_PHEROMONE_DIFFUSION_RATE 0.0001f
#define REPULSIVE_PHEROMONE_SCALE 1.5f
#define REPULSIVE_PHEROMONE_STRENGTH (-0.00000111f)
#define REPULSIVE_PHEROMONE_DECAY_RATE 0.001f
#define REPULSIVE_PHEROMONE_SECRETION_RATE 0.00001f
#define REPULSIVE_PHEROMONE_DIFFUSION_RATE 0.0001f
#define MAXIMUM_AGENTS_PER_CELL 4

__constant__ float DX = WIDTH/N;

struct Agent {
    float x, y, angle, speed;  // Position in 2D space
};

// Function to compute the gradient in the X direction (partial derivative)
__device__ float gradientX(float* grid, int i, int j) {
    //periodic boundary conditions
    int leftIndex = i - 1;
    if (leftIndex < 0) leftIndex += N;
    int rightIndex = i + 1;
    if (rightIndex >= N) rightIndex -= N;
    float left = grid[leftIndex * N + j];
    float right = grid[rightIndex * N + j];

    return (right - left) / (2.0f * DX);  // Central difference
}

// Function to compute the gradient in the Y direction (partial derivative)
__device__ float gradientY(float* grid, int i, int j) {
    int downIndex = j - 1;
    if (downIndex < 0) downIndex += N;
    int upIndex = j + 1;
    if (upIndex >= N) upIndex -= N;
    float down = grid[i * N + downIndex];
    float up = grid[i * N + upIndex];

    return (up - down) / (2.0f * DX);  // Central difference
}

// Function to compute the Laplacian (second derivative)
__device__ float laplacian(float* grid, int i, int j) {
    float center = grid[i * N + j];
    int leftIndex = i - 1;
    if (leftIndex < 0) leftIndex += N;
    int rightIndex = i + 1;
    if (rightIndex >= N) rightIndex -= N;
    int downIndex = j - 1;
    if (downIndex < 0) downIndex += N;
    int upIndex = j + 1;
    if (upIndex >= N) upIndex -= N;
    float left = grid[leftIndex * N + j];
    float right = grid[rightIndex * N + j];
    float down = grid[i * N + downIndex];
    float up = grid[i * N + upIndex];

    float laplacian = (left + right + up + down - 4.0f * center) / (DX * DX);
    if (isnan(laplacian) || isinf(laplacian)) {
        printf("Invalid laplacian %f at (%d, %d)\n", laplacian, i, j);
        printf("Center %f\n", center);
        printf("Left %f\n", left);
        printf("Right %f\n", right);
        printf("Down %f\n", down);
        printf("Up %f\n", up);
    }
    return laplacian;
}


// CUDA kernel to initialize the position of each agent
__global__ void initAgents(Agent* agents, curandState* states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < WORM_COUNT) {
        curand_init(seed, id, 0, &states[id]);
        //agents[id].x = curand_uniform(&states[id]) * WIDTH;
        //agents[id].y = curand_uniform(&states[id]) * HEIGHT;
        //initialise in a random position inside the square centered at WIDTH/4, HEIGHT/4 with side length DX*INITIAL_AREA_NUMBER_OF_CELLS
        agents[id].x = WIDTH/4 + curand_uniform(&states[id]) * DX*INITIAL_AREA_NUMBER_OF_CELLS;
        agents[id].y = HEIGHT/4 + curand_uniform(&states[id]) * DX*INITIAL_AREA_NUMBER_OF_CELLS;
        agents[id].angle = curand_uniform(&states[id]) * 2 * M_PI;
        agents[id].speed = SPEED;
    }
}

// CUDA kernel to initialize the chemical grid concentration
__global__ void initGrid(float* grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        //place 100 units of chemical in the square in the middle of the grid with length 20
        if (i >= N / 2 - 10 && i < N / 2 + 10 && j >= N / 2 - 10 && j < N / 2 + 10) {
            grid[i * N + j] = MAX_CONCENTRATION;
        } else{
            grid[i * N + j] = 0.0f;
        }
    }
}



// CUDA kernel to initialize the pheromone grids
__global__ void initAttractiveAndRepulsivePheromoneGrid(float* attractive_pheromone, float* repulsive_pheromone, float* agent_density_grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        attractive_pheromone[i * N + j] = ATTRACTANT_PHEROMONE_SECRETION_RATE * ATTRACTANT_PHEROMONE_DECAY_RATE * agent_density_grid[i * N + j];
        repulsive_pheromone[i * N + j] = REPULSIVE_PHEROMONE_SECRETION_RATE * REPULSIVE_PHEROMONE_DECAY_RATE * agent_density_grid[i * N + j];
    }
}

//CUDA kernel to initialise the agent count grid
__global__ void initAgentDensityGrid(float* agent_count_grid, Agent* agents){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        agent_count_grid[i * N + j] = 0.0f;
        for (int k = 0; k < WORM_COUNT; ++k) {
            int agent_x = (int)(agents[k].x / DX);
            int agent_y = (int)(agents[k].y / DX);
            if (agent_x == i && agent_y == j) {
                agent_count_grid[i * N + j] += 1.0f;
            }
        }
    }
}

// CUDA kernel to update the position of each agent
__global__ void moveAgents(Agent* agents, curandState* states, float* grid, float*potential, float* agent_count_grid) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < WORM_COUNT) {
        //find the highest concentration of the chemical in the sensing range
        float max_concentration = 0.0f;
        int max_concentration_x = 0;
        int max_concentration_y = 0;

        int agent_x= (int)(agents[id].x / DX);
        int agent_y = (int)(agents[id].y / DX);
        for (int i = -SENSING_RANGE; i <= SENSING_RANGE; ++i) {
            for (int j = -SENSING_RANGE; j <= SENSING_RANGE; ++j) {
                float concentration = 0.0f;
                int xIndex = agent_x+i;
                int yIndex = agent_y+j;
                //apply periodic boundary conditions
                if (xIndex < 0) xIndex += N;
                if (xIndex >= N) xIndex -= N;
                if (yIndex < 0) yIndex += N;
                if (yIndex >= N) yIndex -= N;
                if (xIndex >= 0 && xIndex < N && yIndex >= 0 && yIndex < N) {
                    concentration = grid[xIndex * N + yIndex];
                }

                if (concentration > max_concentration) {
                    max_concentration = concentration;
                    max_concentration_x = i;
                    max_concentration_y = j;

                }
            }
        }
        float bias = atan2((float)max_concentration_y, (float)max_concentration_x );
        float random_angle = curand_uniform(&states[id]) * 2.0f * M_PI;
        float new_direction_x = cosf(random_angle)+(DRIFT_FACTOR * max_concentration*cosf(bias));
        float new_direction_y = sinf(random_angle)+(DRIFT_FACTOR * max_concentration*sinf(bias));
        float fx = LAMBDA * cosf(agents[id].angle) + (1.0f - LAMBDA) * new_direction_x;
        float fy = LAMBDA * sinf(agents[id].angle) + (1.0f - LAMBDA) * new_direction_y;
        float len = sqrt(fx * fx + fy * fy);
        fx /= len;
        fy /= len;
        float new_angle = atan2(fy, fx);
        float new_speed_x = SPEED + curand_uniform(&states[id]) * SIGMA;
        float new_speed_y = SPEED + curand_uniform(&states[id]) * SIGMA;
        if(max_concentration>ODOR_THRESHOLD){// || sensed_odor<ODOR_THRESHOLD){
            float potential_x = gradientX(potential, agent_x, agent_y);
            float potential_y = gradientY(potential, agent_x, agent_y);
            //printf("Potential x: %f, Potential y: %f\n", potential_x, potential_y);
            //printf("Sensed odor: %f\n", sensed_odor);
            new_speed_x =  abs(potential_x) + curand_uniform(&states[id]) * SIGMA;
            new_speed_y = abs(potential_y) + curand_uniform(&states[id]) * SIGMA;
            //printf("new speed x: %f, new speed y: %f\n", new_speed_x, new_speed_y);

        }

        float dx = fx * new_speed_x;
        float dy = fy * new_speed_y;


        agents[id].x += dx;
        agents[id].y += dy;
        agents[id].angle = new_angle;
        //agents[id].speed = new_speed;
        // Apply periodic boundary conditions
        if (agents[id].x < 0) agents[id].x += WIDTH;
        if (agents[id].x >= WIDTH) agents[id].x -= WIDTH;
        if (agents[id].y < 0) agents[id].y += HEIGHT;
        if (agents[id].y >= HEIGHT) agents[id].y -= HEIGHT;
        int new_x = (int)(agents[id].x / DX);
        int new_y = (int)(agents[id].y / DX);

        // Check if the new cell is full
        if (agent_count_grid[new_x * N + new_y] >= MAXIMUM_AGENTS_PER_CELL) {
            // Create an array of indices representing the neighboring cells
            int indices[] = {0, 1, 2, 3};
            // Shuffle the array of indices
            for (int k = MAXIMUM_AGENTS_PER_CELL-1; k > 0; --k) {
                int l = curand(&states[id]) % (k + 1);
                int tmp = indices[k];
                indices[k] = indices[l];
                indices[l] = tmp;
            }
            // Find a non-full neighboring cell
            int dx[] = {-1, 1, 0, 0};
            int dy[] = {0, 0, -1, 1};
            for (int k = 0; k < MAXIMUM_AGENTS_PER_CELL; ++k) {
                int nx = new_x + dx[indices[k]];
                int ny = new_y + dy[indices[k]];
                // Apply periodic boundary conditions
                if (nx < 0) nx += N;
                if (nx >= N) nx -= N;
                if (ny < 0) ny += N;
                if (ny >= N) ny -= N;
                // If the neighboring cell is not full, move the agent to this cell
                if (agent_count_grid[nx * N + ny] < MAXIMUM_AGENTS_PER_CELL) {
                    new_x = nx;
                    new_y = ny;
                    break;
                }
            }
        }

        // If the agent has moved to a new cell, update the agent_count_grid
        if (agent_x != new_x || agent_y != new_y) {
            // Decrease the count in the old cell
            atomicAdd(&agent_count_grid[agent_x * N + agent_y], -1);

            // Increase the count in the new cell
            atomicAdd(&agent_count_grid[new_x * N + new_y], 1);
        }
    }
}

//CUDA kernel to update all the grids (except the potential and the agent count grid)
__global__ void updateGrids(float* grid, float* attractive_pheromone, float* repulsive_pheromone, float* agent_count_grid){
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

        //update attractive pheromone
        laplacian_value = laplacian(attractive_pheromone, i, j);
        float new_attractive_pheromone = attractive_pheromone[i * N + j] + DT * (ATTRACTANT_PHEROMONE_DIFFUSION_RATE * laplacian_value - ATTRACTANT_PHEROMONE_DECAY_RATE * attractive_pheromone[i * N + j] + ATTRACTANT_PHEROMONE_SECRETION_RATE * agent_count_grid[i * N + j] / (DX * DX));
        if (new_attractive_pheromone < 0) new_attractive_pheromone = 0.0f;
        attractive_pheromone[i * N + j] = new_attractive_pheromone;

        //update repulsive pheromone
        laplacian_value = laplacian(repulsive_pheromone, i, j);
        float new_repulsive_pheromone = repulsive_pheromone[i * N + j] + DT * (REPULSIVE_PHEROMONE_DIFFUSION_RATE * laplacian_value - REPULSIVE_PHEROMONE_DECAY_RATE * repulsive_pheromone[i * N + j] + REPULSIVE_PHEROMONE_SECRETION_RATE * agent_count_grid[i * N + j] / (DX * DX));
        if (new_repulsive_pheromone < 0) new_repulsive_pheromone = 0.0f;
        repulsive_pheromone[i * N + j] = new_repulsive_pheromone;
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
__global__ void updatePotential(float* potential, float* grid, float* attractive_pheromone, float* repulsive_pheromone){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        float potential_odor, potential_attractive_pheromone, potential_repulsive_pheromone;
        potential_odor = ATTRACTION_STRENGTH * log10(ATTRACTION_SCALE + grid[i * N + j]);
        potential_attractive_pheromone = ATTRACTANT_PHEROMONE_STRENGTH * log10(ATTRACTANT_PHEROMONE_SCALE + attractive_pheromone[i * N + j]);
        potential_repulsive_pheromone = REPULSIVE_PHEROMONE_STRENGTH * log10(REPULSIVE_PHEROMONE_SCALE + repulsive_pheromone[i * N + j]);
        potential[i * N + j] = potential_odor + potential_attractive_pheromone + potential_repulsive_pheromone;
    }
}

// Function to save the positions of agents in a JSON file
void saveToJSON(const char* filename, Agent* h_agents, int step) {
    static json log;
    static bool initialized = false;

    if (!initialized) {
        // Log simulation parameters only once
        log["parameters"] = {{"WIDTH", WIDTH}, {"HEIGHT", HEIGHT}, {"N", WORM_COUNT}, {"LOGGING_INTERVAL", LOGGING_INTERVAL}, {"N_STEPS", N_STEPS} };
        initialized = true;
    }

    for (int i = 0; i < WORM_COUNT; ++i) {
        log[std::to_string(i)].push_back({ h_agents[i].x, h_agents[i].y });
    }

    std::ofstream outFile(filename);
    outFile << log.dump();  // Pretty-print JSON with an indentation of 4 spaces
    outFile.close();

}

// function to save the grid to a file
void saveGridToJSON(const char* filename, float* h_grid) {
    static json log;
    static bool initialized = false;

    if (!initialized) {
        // Log simulation parameters only once
        log["parameters"] = {{"WIDTH",            WIDTH},
                             {"HEIGHT",           HEIGHT},
                             {"N",                WORM_COUNT},
                             {"LOGGING_INTERVAL", LOGGING_INTERVAL},
                             {"N_STEPS",          N_STEPS}};
        initialized = true;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            //use (i, j) as the key for the JSON object
            log[std::to_string(i)+","+std::to_string(j)].push_back({h_grid[i * N + j]});
        }
    }

    std::ofstream outFile(filename);
    outFile << log.dump(4);  // Pretty-print JSON with an indentation of 4 spaces
    outFile.close();
}

// Function to log the matrix to a file
void logMatrixToFile(const char* filename, float* matrix, int width, int height, int step) {
    std::ofstream outFile(filename + std::to_string(step) + ".txt");
    if (!outFile) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            outFile << matrix[y * width + x] << " ";
        }
        outFile << "\n";
    }

    outFile.close();
}

int main() {
    Agent* d_agents;
    Agent* h_agents = new Agent[WORM_COUNT];
    curandState* d_states;
    bool broken = false;
    size_t size = WORM_COUNT * sizeof(Agent);
    //float target_x = WIDTH / 2;
    //float target_y = HEIGHT / 2;
    float* grid;
    float* h_grid = new float[N * N];

    float* attractive_pheromone;
    float* repulsive_pheromone;
    float* agent_count_grid;
    float* h_attractive_pheromone = new float[N * N];
    float* h_repulsive_pheromone = new float[N * N];
    float* h_agent_count_grid = new float[N * N];


    float* h_potential = new float[N * N];
    float* potential;


    cudaMalloc(&d_agents, size);
    cudaMalloc(&d_states, WORM_COUNT * sizeof(curandState));
    cudaMalloc(&grid, N*N*sizeof(float));
    cudaMalloc(&potential, N*N*sizeof(float));

    // Initialize agent positions and random states
    initAgents<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, time(NULL));
    printf("Initializing agents\n");

    cudaDeviceSynchronize();
    cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    //initialize the agent count grid
    cudaMalloc(&agent_count_grid, N*N*sizeof(float));
    initAgentDensityGrid<<<gridSize, blockSize>>>(agent_count_grid, d_agents);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in initAgentDensityGrid: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(float), cudaMemcpyDeviceToHost);


// Initialize the chemical grid concentration
    initGrid<<<gridSize, blockSize>>>(grid);

// Check for errors in the kernel launch
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

    updatePotential<<<gridSize, blockSize>>>(potential, grid, attractive_pheromone, repulsive_pheromone);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in updatePotential: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_potential, potential, N * N * sizeof(float), cudaMemcpyDeviceToHost);


    //print the grid:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (h_grid[i * N + j] > 0) {
                printf("X ");
            } else {
                printf("  ");
            }

        }
        printf("\n");
    }

    // Move agents in a loop
    for (int i = 0; i < N_STEPS; ++i) {
        printf("Step %d\n", i);

        moveAgents<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, grid, potential, agent_count_grid);
        // Check for errors in the kernel launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        // Copy data from device to host
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(float), cudaMemcpyDeviceToHost);

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
        updatePotential<<<gridSize, blockSize>>>(potential, grid, attractive_pheromone, repulsive_pheromone);
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
                }
            }
        }
        if (broken) {
            break;
        }
        // Save positions to JSON every LOGGING_INTERVAL steps
        if (i % LOGGING_INTERVAL == 0) {
            saveToJSON("/home/nema/CLionProjects/untitled/agents_log.json", h_agents, i);
            //saveGridToJSON("/home/nema/CLionProjects/untitled/grid_log.json", h_grid);
            //saveGridToJSON("/home/nema/CLionProjects/untitled/agent_count_grid.json", h_agent_count_grid);
            logMatrixToFile("/home/nema/CLionProjects/untitled/logs/agent_count/agents_log_step_", h_agent_count_grid, N, N, i);
            logMatrixToFile("/home/nema/CLionProjects/untitled/logs/chemical_concentration/chemical_concentration_step_", h_grid, N, N, i);
            logMatrixToFile("/home/nema/CLionProjects/untitled/logs/attractive_pheromone/attractive_pheromone_step_", h_attractive_pheromone, N, N, i);
            logMatrixToFile("/home/nema/CLionProjects/untitled/logs/repulsive_pheromone/repulsive_pheromone_step_", h_repulsive_pheromone, N, N, i);
            logMatrixToFile("/home/nema/CLionProjects/untitled/logs/potential/potential_step_", h_potential, N, N, i);

        }
    }

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
    return 0;
}
