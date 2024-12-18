//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_INIT_ENV_H
#define UNTITLED_INIT_ENV_H
#include <cuda_runtime.h>
#include <random>
#include "../include/json.hpp"
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

struct ExplorationState{
    ExplorationState() :
            id(-1),  // Initialize to an invalid state
            speed_scale(0.0f),
            speed_spread(0.0f),
            angle_mu(0.0f),
            angle_kappa(0.0f),
            duration_mu(0.0f),
            duration_sigma(0.0f),
            timesteps_in_state(0),
            duration(0),
            max_duration(0),
            angle_mu_sign(1)
    {
        // Explicitly zero out probabilities
        for(int i = 0; i < N_STATES; i++) {
            probabilities[i] = 0.0f;
        }
    }
    int id, timesteps_in_state;
    int duration, max_duration;
    float speed_scale, speed_spread;
    float angle_mu, angle_kappa;
    float duration_mu, duration_sigma;
    float probabilities[N_STATES];
    int angle_mu_sign;
};


struct Agent {
    float x, y, angle, speed, previous_potential, cumulative_potential;  // Position in 2D space
    int state;  // State of the agent: -1 stopped, 0 moving, 1 pirouette
    int is_agent_in_target_area;
    int first_timestep_in_target_area, steps_in_target_area;
    int substate, previous_substate;
    bool is_exploring;
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

void set_probabilities(ExplorationState* state, int timestep){
    if (state == nullptr) {
        std::cerr << "Error: Null state pointer" << std::endl;
        return;
    }

    // Ensure id is set before using it
    if (state->id < 0 || state->id >= N_STATES) {
        std::cerr << "Error: Invalid state ID" << std::endl;
        return;
    }
    //float probability_stddev = 0.01f;
    float negative_correlation =     0.1f;
    float positive_correlation =     0.9f;
    float non_correlated =           0.5f;
    float tau=38.0f;//38.0f;
    float pirouette_probability =   fmaxf((-0.002f *((float) timestep) + 2.5f)/tau, 0.25f/tau);// +
    ;//get_acceptable_white_noise(0.0f, probability_stddev);     //2.5 pirouettes per 2 minutes
    float omega_probability =       1.25f/tau;//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);;                                           //1.25 omega turn per 2 minutes
    float reverse_probability =     fmaxf((-0.001f *((float) timestep)+ 1.6f)/tau, 0.5f/tau);//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);    //1.5 reversals per 2 minutes
    float pause_probability =       0.4f/tau;//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);                                              //0.4 pauses per 2 minutes
    float loop_probability =        0.25f/(tau);//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);                                           //0.25 loops per 2 minutes
    float arc_probability =         0.75f/(tau);//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);                                            //0.75 arcs per 2 minutes
    float line_probability =        fmaxf((-0.001f*((float) timestep)+ 2.5f)/(tau), 0.5f/(tau));//+
    ;//get_acceptable_white_noise(0.0f, probability_stddev);    //2.5 lines per 2 minutes


    switch(state->id) {
        case 0: {
            /*if(state->timesteps_in_state == 0){
                state->cur_lambda = 1.0f/get_acceptable_white_noise(80.0f, 1.0f, 200.0f);
            }*/
            //state->probabilities[0] = positive_correlation * loop_probability;//sample_from_exponential(state->cur_lambda);
            state->probabilities[3] = negative_correlation * pirouette_probability;
            state->probabilities[4] = positive_correlation * omega_probability;
            state->probabilities[5] = non_correlated * reverse_probability;
            state->probabilities[6] = non_correlated * pause_probability;
            break;
        }
        case 1: {
            /*if(state->timesteps_in_state == 0){
                state->cur_lambda = 1.0f/get_acceptable_white_noise(50.0f, 10.0f, 120.0f);
            }*/
            //state->probabilities[1] = positive_correlation * arc_probability; //sample_from_exponential(state->cur_lambda);
            state->probabilities[3] = non_correlated * pirouette_probability;
            state->probabilities[4] = positive_correlation * omega_probability;
            state->probabilities[5] = negative_correlation * reverse_probability;
            state->probabilities[6] = non_correlated * pause_probability;
            break;
        }
        case 2: {
            /*
            if(state->timesteps_in_state == 0){
                state->cur_lambda = 1.0f/get_acceptable_white_noise(30.0f, 5.0f, 60.0f);
            }*/
            //state->probabilities[2] = positive_correlation * line_probability;// sample_from_exponential(state->cur_lambda);
            state->probabilities[3] = non_correlated * pirouette_probability;
            state->probabilities[4] = negative_correlation * omega_probability;
            state->probabilities[5] = non_correlated * reverse_probability;
            state->probabilities[6] = non_correlated * pause_probability;
            break;
        }
        default: {
            state->probabilities[0] = non_correlated * loop_probability;
            state->probabilities[1] = non_correlated * arc_probability;
            state->probabilities[2] = non_correlated * line_probability;
            state->probabilities[3] = non_correlated * pirouette_probability;
            state->probabilities[4] = non_correlated * omega_probability;
            state->probabilities[5] = non_correlated * reverse_probability;
            state->probabilities[6] = non_correlated * pause_probability;
            //avoid self loops for now
            state->probabilities[state->id] = 0.0f;
            break;
        }
    }

    // Normalize probabilities
    float total_prob = 0.0f;
    for (int j = 0; j < N_STATES; j++) {
        //check for negative values, set to 0
        if(state->probabilities[j] < 0){
            state->probabilities[j] = 0.0f;
//printf("Negative probability from state %d to state %d at time step %d\n", state->id, j, timestep);
        }/* else if (state->probabilities[j] > 1.0f) {
            state->probabilities[j] = 1.0f;
        }*/
        total_prob += state->probabilities[j];
    }

    if (total_prob > 0.0f) {
        for (int j = 0; j < N_STATES; j++) {
            state->probabilities[j] /= total_prob;
        }
    }
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


int initProbabilitiesWithParams(ExplorationState* states, Parameters* params) {
    //printf("Initializing probabilities\n");

    for (int i = 0; i < WORM_COUNT; i++) {
        //printf("Worm id: %d\n", i);
        for (int j = 0; j < N_STATES; j++) {
            //printf("State id: %d\n", j);
            // Explicitly initialize each exploration state
            states[i*N_STATES + j] = ExplorationState();
            states[i*N_STATES + j].id = j;

            states[i*N_STATES + j].duration_mu = LOOP_TIME_MU;
            states[i*N_STATES + j].duration_sigma = LOOP_TIME_SIGMA;
            states[i*N_STATES + j].duration = 0;


            states[i*N_STATES + j].angle_mu = params->mus[i*N_STATES + j];
            states[i*N_STATES + j].angle_kappa = params->kappas[i*N_STATES + j];
            states[i*N_STATES + j].speed_scale = params->scales[i*N_STATES + j];
            states[i*N_STATES + j].speed_spread = params->sigmas[i*N_STATES + j];

            /*if(j<3 || j==5){
                states[i*N_STATES + j].speed_scale = OFF_FOOD_SPEED_SCALE_FAST;
                states[i*N_STATES + j].speed_spread = OFF_FOOD_SPEED_SHAPE_FAST;
            }*/
            states[i*N_STATES + j].max_duration = N_STEPS;
            switch(j){
                case 0:

                    states[i*N_STATES + j].duration_mu = params->loop_time_mu;
                    states[i*N_STATES + j].duration_sigma = params->loop_time_sigma;


                    break;
                case 1:
                    states[i*N_STATES + j].duration_mu = params->arc_time_mu;
                    states[i*N_STATES + j].duration_sigma = params->arc_time_sigma;

                    break;
                case 2:
                    states[i*N_STATES + j].duration_mu = params->line_time_mu;
                    states[i*N_STATES + j].duration_sigma = params->line_time_sigma;

            }
        }
        //printf("Setting up durations\n");
        // Set probabilities for each state after initialization
        for (int j = 0; j < N_STATES; j++) {
            //printf("State id: %d\n", j);
            if(j<3) {
                int acceptable_duration = get_duration(states[i * N_STATES + j].duration_mu, states[i * N_STATES + j].duration_sigma, states[i * N_STATES + j].max_duration);
                if (acceptable_duration > 0.0f) {states[i * N_STATES + j].duration = acceptable_duration;}
                else {
                    printf("Could not find acceptable duration for state %d: %d\n", j, states[i * N_STATES + j].duration);
                    printf("parameters: MU %f   SIGMA %f    duration %d\n", states[i * N_STATES + j].duration_mu, states[i * N_STATES + j].duration_sigma, states[i * N_STATES + j].max_duration);
                    return -1;
                }

            }
            set_probabilities(&states[i*N_STATES + j], 0);
        }
    }
    return 0;
}


void initProbabilities(ExplorationState* states) {
    printf("Initializing probabilities\n");

    for (int i = 0; i < WORM_COUNT; i++) {

        for (int j = 0; j < N_STATES; j++) {

            // Explicitly initialize each exploration state
            states[i*N_STATES + j] = ExplorationState();
            states[i*N_STATES + j].id = j;

            if(j<3 || j==5){ //crawling states
                states[i*N_STATES + j].speed_scale = OFF_FOOD_SPEED_SCALE_FAST;
                states[i*N_STATES + j].speed_spread = OFF_FOOD_SPEED_SHAPE_FAST;

            }else{ //turning states
                states[i*N_STATES + j].speed_scale = OFF_FOOD_SPEED_SCALE_SLOW;
                states[i*N_STATES + j].speed_spread = OFF_FOOD_SPEED_SHAPE_SLOW;
            }
            states[i*N_STATES + j].duration_mu = LOOP_TIME_MU;
            states[i*N_STATES + j].duration_sigma = LOOP_TIME_SIGMA;
            states[i*N_STATES + j].duration = 0;

            //states[i*N_STATES + j].max_duration = 0;
            switch(j){
                case 0:
                    states[i*N_STATES + j].duration_mu = LOOP_TIME_MU;
                    states[i*N_STATES + j].duration_sigma = LOOP_TIME_SIGMA;
                    states[i*N_STATES + j].max_duration = 200;
                    states[i*N_STATES + j].angle_mu = M_PI/12;
                    states[i*N_STATES + j].angle_kappa = 15.0f;
                    break;
                case 1:
                    states[i*N_STATES + j].duration_mu = ARC_TIME_MU;
                    states[i*N_STATES + j].duration_sigma = ARC_TIME_SIGMA;
                    states[i*N_STATES + j].max_duration = 140;
                    states[i*N_STATES + j].angle_mu = M_PI/12;
                    states[i*N_STATES + j].angle_kappa = 10.0f;
                    break;
                case 2:
                    states[i*N_STATES + j].duration_mu = LINE_TIME_MU;
                    states[i*N_STATES + j].duration_sigma = LINE_TIME_SIGMA;
                    states[i*N_STATES + j].max_duration = 60;
                    states[i*N_STATES + j].angle_mu = 0;
                    states[i*N_STATES + j].angle_kappa = 15.0f;
                    break;
                case 3:
                    states[i*N_STATES + j].angle_mu = 3*M_PI/4;
                    states[i*N_STATES + j].angle_kappa = 2.0f;
                    break;
                case 4:
                    states[i*N_STATES + j].angle_mu = M_PI/2;
                    states[i*N_STATES + j].angle_kappa = 0.5f;
                    break;
                case 5:
                    states[i*N_STATES + j].angle_mu = 0.0f;
                    states[i*N_STATES + j].angle_kappa = 0.75f;
                    break;
                case 6:
                    states[i*N_STATES + j].angle_mu = 0.0f;
                    states[i*N_STATES + j].angle_kappa = 2.0f;
                    break;
            }
        }

        // Set probabilities for each state after initialization
        for (int j = 0; j < N_STATES; j++) {
            if(j<3) {
                states[i * N_STATES + j].duration = get_duration(states[i * N_STATES + j].duration_mu, states[i * N_STATES + j].duration_sigma, states[i * N_STATES + j].max_duration);
                //  printf("Duration for state %d: %d\n", j, states[i * N_STATES + j].duration);
            }
            set_probabilities(&states[i*N_STATES + j], 0);
        }
    }

}

void updateProbabilities(ExplorationState* states, int timestep){
    for (int i = 0; i < WORM_COUNT; i++) {
        for (int j = 0; j < N_STATES; j++) {
            set_probabilities(&states[i*N_STATES + j], timestep);
        }
    }
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
            //initialise in a random position inside the square centered at WIDTH/4, HEIGHT/4 with side length DX*INITIAL_AREA_NUMBER_OF_CELLS
           agents[id].x = WIDTH / 4 - INITIAL_AREA_NUMBER_OF_CELLS/2 * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
           agents[id].y = HEIGHT / 2 - INITIAL_AREA_NUMBER_OF_CELLS/2  * DX + curand_uniform(&states[id]) * INITIAL_AREA_NUMBER_OF_CELLS * DX;
        }
        //generate angle in the range [-pi, pi]
        agents[id].angle =(2.0f * curand_uniform(&states[id]) - 1.0f) * M_PI;
        agents[id].speed = SPEED;
        agents[id].state = 1;
        agents[id].previous_potential = 0.0f;
        agents[id].cumulative_potential = 0.0f;
        agents[id].is_agent_in_target_area = 0;
        agents[id].first_timestep_in_target_area = -1;
        agents[id].steps_in_target_area = 0;
        agents[id].is_exploring = true;
        agents[id].substate = (int) curand_uniform(&states[id]) * N_STATES;
        agents[id].previous_substate = 0;

    }
}

// CUDA kernel to initialize the chemical grid concentration
__global__ void initGrid(float* grid, curandState* states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        //place 100 units of chemical in the square in the middle of the grid with length 20
        if (i >= 3 * N / 4 -  TARGET_AREA_SIDE_LENGTH/2 && i < 3 * N / 4 + TARGET_AREA_SIDE_LENGTH/2 && j >= N / 2 - TARGET_AREA_SIDE_LENGTH/2 && j < N / 2 + TARGET_AREA_SIDE_LENGTH/2) {
        //if (i >=N / 2 - TARGET_AREA_SIDE_LENGTH/2 && i <  N / 2 + TARGET_AREA_SIDE_LENGTH/2 && j >= N / 2 - TARGET_AREA_SIDE_LENGTH/2 && j < N / 2 + TARGET_AREA_SIDE_LENGTH/2) {
            grid[i * N + j] = MAX_CONCENTRATION * (1.0f + curand_normal(&states[i*N+j]));
        } else{
            grid[i * N + j] = 0.0f;
        }
    }
}

// CUDA kernel to initialize the chemical grid concentration in an approximated circle of radius TARGET_AREA_SIDE_LENGTH/2
__global__ void initGridWithCircle(float* grid, curandState* states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < N) {
        //place 100 units of chemical in the circle in the middle of the grid with radius 20
        if ((i - 3 * N / 4) * (i - 3 * N / 4) + (j - N / 2) * (j - N / 2) <= (TARGET_AREA_SIDE_LENGTH / 2) * (TARGET_AREA_SIDE_LENGTH / 2)) {
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
