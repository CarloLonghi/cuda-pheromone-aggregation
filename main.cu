#include <stdio.h>
#include <curand_kernel.h>
#include "include/json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

#define N 128                        // Grid size
#define WORM_COUNT 5                 // Number of agents
#define WIDTH 60.0f          // Width of the 2D space
#define HEIGHT 60.0f         // Height of the 2D space
#define BLOCK_SIZE 32        // CUDA block size
#define N_STEPS 3600         // Number of simulation steps
#define LOGGING_INTERVAL 1    // Logging interval for saving positions
#define SPEED 0.1f            // Constant speed at which agents move
//#define DX WIDTH/N               // Grid spacing
#define LAMBDA 0.8f             //persistance of the movement
#define DRIFT_FACTOR 0.01f       //drift factor
#define SENSING_RANGE 1        //sensing range of the agents
#define MAX_CONCENTRATION 10.0f //maximum concentration of the chemical
#define DT 0.1f              //time step
#define GAMMA 0.001f             //decay rate of the chemical
#define DIFFUSION_CONSTANT 0.05f                  //diffusion rate of the chemical
#define ATTRACTION_STRENGTH 0.0282f
#define ATTRACTION_SCALE 15.0f
#define ODOR_THRESHOLD 1e-6
#define DEBUG false
#define SIGMA 0.00027f
#define INITIAL_AREA_NUMBER_OF_CELLS 20 //defines the side length of the square where the agents are initialized in terms of number of cells
//pheromone parameters
#define ATTRACTANT_PHEROMONE_SCALE 15.0f
#define ATTRACTANT_PHEROMONE_STRENGTH 0.000282f
#define ATTRACTANT_PHEROMONE_DECAY_RATE 0.01f
#define ATTRACTANT_PHEROMONE_SECRETION_RATE 0.0001f
#define ATTRACTANT_PHEROMONE_DIFFUSION_RATE 0.0001f
#define REPULSIVE_PHEROMONE_SCALE 15.0f
#define REPULSIVE_PHEROMONE_STRENGTH (-0.0000031f)
#define REPULSIVE_PHEROMONE_DECAY_RATE 0.001f
#define REPULSIVE_PHEROMONE_SECRETION_RATE 0.00001f
#define MAXIMUM_AGENTS_PER_CELL 400
#define REPULSIVE_PHEROMONE_DIFFUSION_RATE 0.001f
#define PIROUETTE_TO_RUN_THRESHOLD 1e-6f
#define ENABLE_MAXIMUM_NUMBER_OF_AGENTS_PER_CELL false

#define ON_FOOD_AVERAGE_SPEED 0.17f
#define ON_FOOD_SPEED_SIGMA 0.5f
#define OFF_FOOD_AVERAGE_SPEED 0.179f
#define OFF_FOOD_SPEED_SIGMA 0.7f

#define LOG_VELOCITIES false
#define LOG_ANGLES false
#define LOG_POTENTIAL false

__constant__ float DX = WIDTH/N;


struct Agent {
    float x, y, angle, speed, previous_potential;  // Position in 2D space
    int state;  // State of the agent: -1 stopped, 0 moving, 1 pirouette

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
__global__ void initAgents(Agent* agents, curandState* states, unsigned long seed, int worm_count) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {
        curand_init(seed, id, 0, &states[id]);
        //agents[id].x = curand_uniform(&states[id]) * WIDTH;
        //agents[id].y = curand_uniform(&states[id]) * HEIGHT;
        //initialise in a random position inside the square centered at WIDTH/4, HEIGHT/4 with side length DX*INITIAL_AREA_NUMBER_OF_CELLS
        agents[id].x = WIDTH/4 + curand_uniform(&states[id]) * DX*INITIAL_AREA_NUMBER_OF_CELLS;
        agents[id].y = HEIGHT/2 + curand_uniform(&states[id]) * DX*INITIAL_AREA_NUMBER_OF_CELLS;
        agents[id].angle = curand_uniform(&states[id]) * 2 * M_PI;
        agents[id].speed = SPEED;
        agents[id].state = 0;
        agents[id].previous_potential = 0.0f;
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
        attractive_pheromone[i * N + j] = ATTRACTANT_PHEROMONE_SECRETION_RATE * ATTRACTANT_PHEROMONE_DECAY_RATE * (float)agent_density_grid[i * N + j] / (DX*DX);
        repulsive_pheromone[i * N + j] = REPULSIVE_PHEROMONE_SECRETION_RATE * REPULSIVE_PHEROMONE_DECAY_RATE * (float)agent_density_grid[i * N + j]/ (DX*DX);
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
                agent_count_grid[i * N + j] += 1;
            }
        }
    }
}

// CUDA kernel to update the position of each agent
__global__ void moveAgents(Agent* agents, curandState* states, float* potential, int* agent_count_grid, int worm_count) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {
        //find the highest concentration of the chemical in the sensing range
        // or find the minimum potential in the sensing range
        float max_concentration = 0.0f;
        int max_concentration_x = 0;
        int max_concentration_y = 0;

        int agent_x= (int)(agents[id].x / DX);
        int agent_y = (int)(agents[id].y / DX);
        float sum_of_potentials = 0.0f;
        int n_neighbors = 0;
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
                    concentration = potential[xIndex * N + yIndex];
                    //printf("At (%d, %d) concentration: %f\n", xIndex, yIndex, concentration);
                    sum_of_potentials += concentration;
                    n_neighbors++;
                }

                if (concentration > max_concentration) {
                    max_concentration = concentration;
                    max_concentration_x = i;
                    max_concentration_y = j;

                }
            }
        }
        /*
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
        */
        float sensed_potential = potential[agent_x * N + agent_y];

        if (sensed_potential - agents[id].previous_potential > PIROUETTE_TO_RUN_THRESHOLD){ //starting to move in the "right" direction, then RUN
            agents[id].state = 0;
        }
        else if (sensed_potential - agents[id].previous_potential < 0.0f){ //moving in the wrong direction, then PIROUETTE
            agents[id].state = 1;
        }
        float fx, fy, new_angle, random_angle = (2.0f*curand_uniform(&states[id])-1.0f) * M_PI;
        if(agents[id].state == 0){ //if the agent is moving = RUN - LOW TURNING - LEVY FLIGHT
            float bias = atan2((float)max_concentration_y, (float)max_concentration_x );
            float new_direction_x = cosf(random_angle)+(DRIFT_FACTOR * max_concentration*cosf(bias));
            float new_direction_y = sinf(random_angle)+(DRIFT_FACTOR * max_concentration*sinf(bias));
            fx = LAMBDA * cosf(agents[id].angle) + (1.0f - LAMBDA) * new_direction_x;
            fy = LAMBDA * sinf(agents[id].angle) + (1.0f - LAMBDA) * new_direction_y;
            new_angle = atan2(fy, fx);
            agents[id].angle = new_angle;
        }
        else{ //BROWNIAN MOTION - HIGH TURNING - PIROUETTE
            float random_angle2 = (2.0f *curand_uniform(&states[id]) -1.0f)*M_PI ;
            agents[id].angle += random_angle2;
            if (agents[id].angle > M_PI) agents[id].angle -=  M_PI;
            if (agents[id].angle < -M_PI) agents[id].angle +=  M_PI;
            fx = cosf(agents[id].angle);
            fy = sinf(agents[id].angle);

        }


        float new_speed_x = SPEED;
        float new_speed_y = SPEED;

        if(sensed_potential>ODOR_THRESHOLD){ //on food - lognorm distribution of speed
            //new_speed_x = ON_FOOD_AVERAGE_SPEED + curand_normal(&states[id]) * ON_FOOD_SPEED_SIGMA;
            //new_speed_y = ON_FOOD_AVERAGE_SPEED + curand_normal(&states[id]) * ON_FOOD_SPEED_SIGMA;
            new_speed_x = curand_log_normal(&states[id], logf(ON_FOOD_AVERAGE_SPEED), ON_FOOD_SPEED_SIGMA);
            new_speed_y = curand_log_normal(&states[id], logf(ON_FOOD_AVERAGE_SPEED), ON_FOOD_SPEED_SIGMA);
        }
        else{ //off food - lognormal distribution of speed
            new_speed_x = curand_log_normal(&states[id], logf(OFF_FOOD_AVERAGE_SPEED), OFF_FOOD_SPEED_SIGMA);
            new_speed_y = curand_log_normal(&states[id], logf(OFF_FOOD_AVERAGE_SPEED), OFF_FOOD_SPEED_SIGMA);

        }
        //printf("New speed x: %f, New speed y: %f\n", new_speed_x, new_speed_y);
        float dx = fx * new_speed_x;
        float dy = fy * new_speed_y;

        agents[id].previous_potential = sensed_potential;
        agents[id].x += dx;
        agents[id].y += dy;
        //agents[id].angle = new_angle;
        agents[id].speed = sqrt(dx * dx + dy * dy);
        // Apply periodic boundary conditions
        if (agents[id].x < 0) agents[id].x += WIDTH;
        if (agents[id].x >= WIDTH) agents[id].x -= WIDTH;
        if (agents[id].y < 0) agents[id].y += HEIGHT;
        if (agents[id].y >= HEIGHT) agents[id].y -= HEIGHT;
        int new_x = (int)(agents[id].x / DX);
        int new_y = (int)(agents[id].y / DX);

        if(ENABLE_MAXIMUM_NUMBER_OF_AGENTS_PER_CELL){
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
__global__ void updateGrids(float* grid, float* attractive_pheromone, float* repulsive_pheromone, int* agent_count_grid){
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
__global__ void updatePotential(float* potential, float* grid, float* attractive_pheromone, float* repulsive_pheromone, float attractive_pheromone_strength, float repulsive_pheromone_strength, float odor_strength) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        float potential_odor=0.0f, potential_attractive_pheromone=0.0f, potential_repulsive_pheromone = 0.0f;
        potential_odor = odor_strength * log10(ATTRACTION_SCALE + grid[i * N + j]);// / (ATTRACTION_SCALE + grid[i * N + j]);

        //
         potential_attractive_pheromone = attractive_pheromone_strength * log10(ATTRACTANT_PHEROMONE_SCALE + attractive_pheromone[i * N + j]);// / (ATTRACTANT_PHEROMONE_SCALE + attractive_pheromone[i * N + j]);

        potential_repulsive_pheromone = repulsive_pheromone_strength * log10(REPULSIVE_PHEROMONE_SCALE + repulsive_pheromone[i * N + j]);// / (REPULSIVE_PHEROMONE_SCALE + repulsive_pheromone[i * N + j]);

        potential[i * N + j] = potential_odor + potential_attractive_pheromone + potential_repulsive_pheromone;


    }
}

// Function to save the positions of agents in a JSON file
void saveToJSON(const char* filename, Agent* h_agents, int worm_count, const char* angle_filename, const char* velocity_filename) {
    static json log;
    static bool initialized = false;

    if (!initialized) {
        // Log simulation parameters only once
        log["parameters"] = {{"WIDTH", WIDTH}, {"HEIGHT", HEIGHT}, {"N", worm_count}, {"LOGGING_INTERVAL", LOGGING_INTERVAL}, {"N_STEPS", N_STEPS} };
        initialized = true;
    }

    for (int i = 0; i < worm_count; ++i) {
        log[std::to_string(i)].push_back({ h_agents[i].x, h_agents[i].y });
    }

    std::ofstream outFile(filename);
    outFile << log.dump();  // Pretty-print JSON with an indentation of 4 spaces
    outFile.close();


    if(LOG_ANGLES) {
        //same for the angles
        static json log_angles;
        static bool initialized_angles = false;

        if (!initialized_angles) {
            // Log simulation parameters only once
            log_angles["parameters"] = {{"WIDTH",            WIDTH},
                                        {"HEIGHT",           HEIGHT},
                                        {"N", worm_count},
                                        {"LOGGING_INTERVAL", LOGGING_INTERVAL},
                                        {"N_STEPS",          N_STEPS}};
            initialized_angles = true;
        }

        for (int i = 0; i < worm_count; ++i) {
            log_angles[std::to_string(i)].push_back({h_agents[i].angle});
        }

        std::ofstream outFile_angles(angle_filename);
        outFile_angles << log_angles.dump();  // Pretty-print JSON with an indentation of 4 spaces
        outFile_angles.close();
    }
    //same for velocities
    if(LOG_VELOCITIES) {
        static json log_velocities;
        static bool initialized_velocities = false;

        if (!initialized_velocities) {
            // Log simulation parameters only once
            log_velocities["parameters"] = {{"WIDTH",            WIDTH},
                                            {"HEIGHT",           HEIGHT},
                                            {"N", worm_count},
                                            {"LOGGING_INTERVAL", LOGGING_INTERVAL},
                                            {"N_STEPS",          N_STEPS}};
            initialized_velocities = true;
        }

        for (int i = 0; i < worm_count; ++i) {
            log_velocities[std::to_string(i)].push_back({h_agents[i].speed});
        }

        std::ofstream outFile_velocities(velocity_filename);
        outFile_velocities << log_velocities.dump();  // Pretty-print JSON with an indentation of 4 spaces
        outFile_velocities.close();
    }

}

// function to save the grid to a file
void saveGridToJSON(const char* filename, float* h_grid, int worm_count) {
    static json log;
    static bool initialized = false;

    if (!initialized) {
        // Log simulation parameters only once
        log["parameters"] = {{"WIDTH",            WIDTH},
                             {"HEIGHT",           HEIGHT},
                             {"N",                worm_count},
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

void logIntMatrixToFile(const char* filename, int* matrix, int width, int height, int step) {
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

void savePositionsToJSON(const char* filename, float* positions, int worm_count, int n_steps) {
    json log;

    // Log simulation parameters
    log["parameters"] = {{"WIDTH",            WIDTH},
                         {"HEIGHT",           HEIGHT},
                         {"N", worm_count},
                         {"LOGGING_INTERVAL", LOGGING_INTERVAL},
                         {"N_STEPS",          N_STEPS}};

    // Log positions
    for (int i = 0; i < worm_count; ++i) {
        for (int j = 0; j < n_steps; ++j) {
            log[std::to_string(i)].push_back(
                    {positions[(j * worm_count + i) * 2], positions[(j * worm_count + i) * 2 + 1]});
        }
    }

    std::ofstream outFile(filename);
    outFile << log.dump(4);  // Pretty-print JSON with an indentation of 4 spaces
    outFile.close();
}

int main(int argc, char* argv[]) {
    float attractant_pheromone_strength = ATTRACTANT_PHEROMONE_STRENGTH;
    float repulsive_pheromone_strength = REPULSIVE_PHEROMONE_STRENGTH;
    int worm_count = WORM_COUNT;
    int exp_number = 0;
    float odor_strength = ATTRACTION_STRENGTH;
    printf("Found %d arguments\n", argc);
    if (argc ==3 && std::isdigit(argv[1][0]) && std::isdigit(argv[2][0])){
        attractant_pheromone_strength = std::stof(argv[1]);
        printf("Attractant pheromone strength: %.10f\n", attractant_pheromone_strength);
        repulsive_pheromone_strength = std::stof(argv[2]);
        printf("Repulsive pheromone strength: %.10f\n", repulsive_pheromone_strength);
    }
    else {
        if(argc == 6 && std::isdigit(argv[1][0]) && std::isdigit(argv[2][0]) && std::isdigit(argv[3][0]) && std::isdigit(argv[4][0])  && std::isdigit(argv[5][0])){
            exp_number = std::stoi(argv[1]);
            printf("Experiment number: %d\n", exp_number);
            worm_count = std::stoi(argv[2]);
            printf("Worm count: %d\n", worm_count);
            attractant_pheromone_strength = std::stof(argv[3]);
            printf("Attractant pheromone strength: %.10f\n", attractant_pheromone_strength);
            repulsive_pheromone_strength = - std::stof(argv[4]);
            printf("Repulsive pheromone strength: %.10f\n", repulsive_pheromone_strength);
            int using_odor = std::stoi(argv[5]);
            if(using_odor == 0){
                odor_strength = 0.0f;
            }
            printf("Odor strength: %f\n", odor_strength);
        }
        else {
            printf("No input arguments provided.\n");
        }
    }

    Agent* d_agents;
    Agent* h_agents = new Agent[worm_count];
    //the following should be the sum of all the maximum values of the potentials. However, we assume the potential of pheromones is always smaller than that of the chemical, so we use that value * 3
    float MAXIMUM_POTENTIAL =  ATTRACTION_STRENGTH * log10(ATTRACTION_SCALE + MAX_CONCENTRATION);
    curandState* d_states;
    bool broken = false;
    size_t size = worm_count * sizeof(Agent);
    //float target_x = WIDTH / 2;
    //float target_y = HEIGHT / 2;
    float* grid;
    float* h_grid = new float[N * N];

    float* attractive_pheromone;
    float* repulsive_pheromone;
    int* agent_count_grid;
    float* h_attractive_pheromone = new float[N * N];
    float* h_repulsive_pheromone = new float[N * N];
    int* h_agent_count_grid = new int[N * N];


    float* h_potential = new float[N * N];
    float* potential;

    float* positions = new float[worm_count * N_STEPS * 2]; // Matrix to store positions (x, y) for each agent at each timestep

    cudaMalloc(&d_agents, size);
    cudaMalloc(&d_states, worm_count * sizeof(curandState));
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
    cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(float), cudaMemcpyDeviceToHost);


// Initialize the chemical grid concentration
    initGrid<<<gridSize, blockSize>>>(grid);
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
    updatePotential<<<gridSize, blockSize>>>(potential, grid, attractive_pheromone, repulsive_pheromone, attractant_pheromone_strength, repulsive_pheromone_strength, odor_strength);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in updatePotential: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_potential, potential, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Move agents in a loop
    for (int i = 0; i < N_STEPS; ++i) {
        //printf("Step %d\n", i);

        moveAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_states, potential, agent_count_grid, worm_count);
        // Check for errors in the kernel launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        // Copy data from device to host
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_agent_count_grid, agent_count_grid, N * N * sizeof(float), cudaMemcpyDeviceToHost);

        // Store positions in the matrix
        for (int j = 0; j < worm_count; ++j) {
            positions[(i * worm_count + j) * 2] = h_agents[j].x;
            positions[(i * worm_count + j) * 2 + 1] = h_agents[j].y;
        }
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
        updatePotential<<<gridSize, blockSize>>>(potential, grid, attractive_pheromone, repulsive_pheromone, attractant_pheromone_strength, repulsive_pheromone_strength, odor_strength);
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
//saveToJSON("/home/nema/CLionProjects/untitled/agents_log.json", h_agents, worm_count, "/home/nema/CLionProjects/untitled/agents_angles_log.json", "/home/nema/CLionProjects/untitled/agents_velocities_log.json");
            //saveGridToJSON("/home/nema/CLionProjects/untitled/grid_log.json", h_grid);
            //saveGridToJSON("/home/nema/CLionProjects/untitled/agent_count_grid.json", h_agent_count_grid);
            //logIntMatrixToFile("/home/nema/CLionProjects/untitled/logs/agent_count/agents_log_step_", h_agent_count_grid, N, N, i);
            //logMatrixToFile("/home/nema/CLionProjects/untitled/logs/chemical_concentration/chemical_concentration_step_", h_grid, N, N, i);
            //logMatrixToFile("/home/nema/CLionProjects/untitled/logs/attractive_pheromone/attractive_pheromone_step_", h_attractive_pheromone, N, N, i);
            //logMatrixToFile("/home/nema/CLionProjects/untitled/logs/repulsive_pheromone/repulsive_pheromone_step_", h_repulsive_pheromone, N, N, i);
            if(LOG_POTENTIAL) {
                logMatrixToFile("/home/nema/CLionProjects/untitled/potential/potential_step_", h_potential, N, N, i);
            }

        }
    }
    savePositionsToJSON("/home/nema/CLionProjects/untitled/agents_log.json", positions, worm_count, N_STEPS);

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
    return 0;
}
