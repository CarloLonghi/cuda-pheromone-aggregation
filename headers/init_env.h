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


struct Parameters{
    Parameters() :
            kappas{0.0f},
            mus{0.0f},
            sigmas{0.0f},
            scales{0.0f}
    {
        // Explicitly zero out parameters
        for(int i = 0; i < N_STATES * WORM_COUNT; i++) {
            kappas[i] = 0.0f;
            mus[i] = 0.0f;
            sigmas[i] = 0.0f;
            scales[i] = 0.0f;
        }
    }

    float kappas[N_STATES * WORM_COUNT];
    float mus[N_STATES * WORM_COUNT];
    float sigmas[N_STATES * WORM_COUNT];
    float scales[N_STATES * WORM_COUNT];
    float loop_time_mu, loop_time_sigma, arc_time_mu, arc_time_sigma, line_time_mu, line_time_sigma;

};


struct Agent {
    float x, y, angle, speed, previous_potential, cumulative_potential;  // Position in 2D space
};

float sample_from_exponential(float lambda){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> d(lambda);
    float sample = d(gen);
    /*while(sample>1.0f){
        sample = d(gen);
    }*/
    return sample;
}

float get_acceptable_white_noise(float mean, float stddev, float bound=1.0f){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, stddev);
    float white_noise;

    white_noise = (float) dist(gen);
    while(abs(white_noise)>bound || white_noise<0){
        white_noise = (float) dist(gen);
    }
    return white_noise;
}

void loadSpeedParameters(Parameters* params, const char* filename){
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    json data = json::parse(file);
    for(int i=0; i<WORM_COUNT; i++) {
        for (int j = 0; j < N_STATES; j++) {
            int base_idx = j * 2;
            params->scales[i*N_STATES+j] = data[base_idx].get<float>();
            params->sigmas[i*N_STATES+j] = data[base_idx + 1].get<float>();
        }

        params->loop_time_mu = data[N_STATES * 2].get<float>();
        params->loop_time_sigma = data[N_STATES * 2 + 1].get<float>();
        params->arc_time_mu = data[N_STATES * 2 + 2].get<float>();
        params->arc_time_sigma = data[N_STATES * 2 + 3].get<float>();
        params->line_time_mu = data[N_STATES * 2 + 4].get<float>();
        params->line_time_sigma = data[N_STATES * 2 + 5].get<float>();
    }
    file.close();
}

void loadBatchSingleAgentParameters(Parameters* params, const char* filename, int id){
    //does the same as the others, but puts only the parameters of agent id in the params struct
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    json data = json::parse(file);

    // Convert agent ID to string to access its array in JSON
    std::string agent_id = std::to_string(id);
    const auto& agent_params = data[agent_id];
    for(int i=0; i<WORM_COUNT; i++){
        for(int j=0; j<N_STATES; j++){
            int base_idx = j*2; // Each state has Mu and Kappa (2 values)
            params->mus[i*N_STATES + j] = agent_params[base_idx].get<float>();
            params->kappas[i*N_STATES + j] = agent_params[base_idx + 1].get<float>();
        }
        // Access time-related parameters after the states
        int time_base_idx = N_STATES*2; // Time parameters start after the states
        params->loop_time_mu = agent_params[time_base_idx].get<float>();
        params->loop_time_sigma = agent_params[time_base_idx + 1].get<float>();
        params->arc_time_mu = agent_params[time_base_idx + 2].get<float>();
        params->arc_time_sigma = agent_params[time_base_idx + 3].get<float>();
        params->line_time_mu = agent_params[time_base_idx + 4].get<float>();
        params->line_time_sigma = agent_params[time_base_idx + 5].get<float>();
    }
    file.close();
}

void loadOptimisedParametersSingleAgent(Parameters* params, const char* filename, int id){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    json data = json::parse(file);
    std::string agent_id = std::to_string(id);
    const auto& agent_params = data[agent_id];
    for(int i=0; i<WORM_COUNT; i++){
        for(int j=0; j<N_STATES; j++){
            int base_idx = j*2; // Each state has Mu and Kappa (2 values)
            params->mus[i*N_STATES + j] = agent_params[base_idx].get<float>();
            params->kappas[i*N_STATES + j] = agent_params[base_idx + 1].get<float>();
        }
    }
    file.close();
}

void loadOptimisedParameters13Agents(Parameters* params, const char* filename){
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Parse the JSON file
    json data = json::parse(file);

    for (int i = 0; i < WORM_COUNT; i++) {
        // Convert agent ID to string to access its array in JSON
        std::string agent_id = std::to_string(i);

        if (!data.contains(agent_id)) {
            std::cerr << "Error: Agent ID " << agent_id << " not found in JSON" << std::endl;
            continue;
        }

        // Access the flat array of parameters for the current agent
        const auto& agent_params = data[agent_id];

        for (int j = 0; j < N_STATES; j++) {
            int base_idx = j * 2; // Each state has Mu and Kappa (2 values)
            params->mus[i * N_STATES + j] = agent_params[base_idx].get<float>();
            params->kappas[i * N_STATES + j] = agent_params[base_idx + 1].get<float>();
        }

        // Access time-related parameters after the states
        int time_base_idx = N_STATES * 2; // Time parameters start after the states
        params->loop_time_mu = agent_params[time_base_idx].get<float>();
        params->loop_time_sigma = agent_params[time_base_idx + 1].get<float>();
        params->arc_time_mu = agent_params[time_base_idx + 2].get<float>();
        params->arc_time_sigma = agent_params[time_base_idx + 3].get<float>();
        params->line_time_mu = agent_params[time_base_idx + 4].get<float>();
        params->line_time_sigma = agent_params[time_base_idx + 5].get<float>();
    }

    file.close();
}



void loadParameters(Parameters* params, const char* filename, bool load_times=true) {
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    json data = json::parse(file);
    //printf("Data: %s\n", data.dump().c_str());
    for (int j = 0; j < N_STATES; j++) {
        int base_idx = j * 2;
        params->mus[j] = data[base_idx].get<float>();
        params->kappas[j] = data[base_idx + 1].get<float>();
    }
    if(load_times) {
        params->loop_time_mu = data[N_STATES * 2].get<float>();
        params->loop_time_sigma = data[N_STATES * 2 + 1].get<float>();
        params->arc_time_mu = data[N_STATES * 2 + 2].get<float>();
        params->arc_time_sigma = data[N_STATES * 2 + 3].get<float>();
        params->line_time_mu = data[N_STATES * 2 + 4].get<float>();
        params->line_time_sigma = data[N_STATES * 2 + 5].get<float>();
    }
    file.close();
}


int get_duration(float mu, float sigma, int upper_bound){
    std::mt19937 engine; // uniform random bit engine

    // seed the URBG
    std::random_device dev{};
    engine.seed(dev());
    std::lognormal_distribution<double> dist(mu, sigma);
    int duration = (int) dist(engine);
    int max_retries = 50;
    while (duration < 0 || duration > upper_bound) {
        printf("duration %f\n", dist(engine));
        duration = (int) dist(engine);
        max_retries -= 1;
        if (max_retries<=0) break;
    }
    if (max_retries <= 0) {duration=-1;}
    return duration;
}


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
    if (i < N && j < N) {
        if(agent_density_grid[i * N + j] == 0){
            attractive_pheromone[i * N + j] = 0.0f;
            repulsive_pheromone[i * N + j] = 0.0f;
        }
        else {
            attractive_pheromone[i * N + j] = attractive_pheromone_secretion_rate * attractive_pheromone_decay_rate *
                                              (float) agent_density_grid[i * N + j] / (DX * DX);
            repulsive_pheromone[i * N + j] = repulsive_pheromone_secretion_rate * repulsive_pheromone_decay_rate *
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
            int agent_x = (int)round(agents[k].x / DX);
            int agent_y = (int)round(agents[k].y / DY);
            if (agent_x == i && agent_y == j) {
                // printf("Agent at (%d, %d)\n", i, j);
                agent_count_grid[i * N + j] += 1;
            }
        }
    }
}
#endif //UNTITLED_INIT_ENV_H
