//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_AGENT_UPDATE_H
#define UNTITLED_AGENT_UPDATE_H
#include <cuda_runtime.h>
#include <random>
#include <limits>
#include <cmath>
#include "gaussian_odour.h"
#include "numeric_functions.h"
// Function to sample from a von Mises distribution
__device__ float sample_from_von_mises(float mu, float kappa, curandState* state) {
    // Handle kappa = 0 (uniform distribution)
    if (kappa < 1e-6) {
        return mu + (2.0f * M_PI * curand_uniform(state)) - M_PI; // Random uniform sample
    }

    // Step 1: Setup variables
    float a = 1.0f + sqrt(1.0f + 4.0f * kappa * kappa);
    float b = (a - sqrt(2.0f * a)) / (2.0f * kappa);
    float r = (1.0f + b * b) / (2.0f * b);

    /*std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    */
    while (true) {
        // Step 2: Generate random variables
        float u1 = abs(curand_uniform(state));
        float z = cos(M_PI * u1);
        float f = (1.0f + r * z) / (r + z);
        float c = kappa * (r - f);

        // Step 3: Generate random variable u2
        float u2 = abs(curand_uniform(state));

        // Step 4: Accept/reject step
        if (u2 < c * (2.0f - c) || u2 <= c * exp(1.0f - c)) {
            // Step 5: Generate final angle sample
            float u3 = abs(curand_uniform(state));
            float theta = (u3 < 0.5f) ? acos(f) : -acos(f);
            float result = mu + theta;  // Return the sample from von Mises
            if (result > M_PI) {
                result -= 2.0f * M_PI;
            } else if (result < -M_PI) {
                result += 2.0f * M_PI;
            }
            return result;
        }
    }
}

__device__ int select_next_state(float* probabilities, curandState* local_state, int num_states) {
    // Generate a random value between 0 and 1
    float random_val = curand_uniform(local_state);

    // Cumulative probability tracking
    float cumulative_prob = 0.0f;

    // Iterate through probabilities to select state
    for (int i = 0; i < num_states; i++) {
        cumulative_prob += probabilities[i];
        // If random value is less than cumulative probability, select this state
        if (random_val <= cumulative_prob) {
            return i;
        }
    }
    printf("Error: No state selected\n");
    // Fallback to last state if no state is selected (should rarely happen)
    return num_states - 1;
}

// CUDA kernel to update the position of each agent
__global__ void moveAgents(Agent* agents, curandState* states, ExplorationState * d_explorationStates,  /*float* potential, int* agent_count_grid,*/ int worm_count, int timestep, float sigma) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {

        float max_concentration_x = 0.0f;
        float max_concentration_y = 0.0f;
        float sensed_potential = computeDensityAtPoint(agents[id].x, agents[id].y, timestep);//potential[agent_x * N + agent_y];
        sensed_potential = ATTRACTION_STRENGTH * logf(sensed_potential + ATTRACTION_SCALE);
        //add a small perceptual noise to the potential
        if(sigma!=0.0f){
            float perceptual_noise = curand_normal(&states[id]) * sigma;
            if(perceptual_noise>sigma) perceptual_noise = sigma;
            if(perceptual_noise<-sigma) perceptual_noise = (-sigma);
            sensed_potential += perceptual_noise;
        }

        float max_concentration = sensed_potential;
        //printf("Sensed potential: %f\n", sensed_potential);
        for (int i = 0; i < 32; ++i) {
            float angle = curand_uniform(&states[id]) * 2 * M_PI;
            float sample_x = agents[id].x + SENSING_RADIUS * cosf(angle);
            float sample_y = agents[id].y + SENSING_RADIUS * sinf(angle);
            float concentration = computeDensityAtPoint(sample_x, sample_y, timestep);
            // Add perceptual noise if sigma is not zero
            if (sigma != 0.0f) {
                concentration += curand_normal(&states[id]) * sigma;
            }
            concentration = ATTRACTION_STRENGTH * logf(concentration + ATTRACTION_SCALE);

            if (concentration > max_concentration) {
                max_concentration = concentration;
                max_concentration_x = cosf(angle);
                max_concentration_y = sinf(angle);
            }
        }

        float auto_transition_probability = curand_uniform(&states[id]);
        if (agents[id].cumulative_potential > PIROUETTE_TO_RUN_THRESHOLD){ //|| auto_transition_probability>=AUTO_TRANSITION_PROBABILITY_THRESHOLD){ //starting to move in the "right" direction, then RUN
            agents[id].state = 0;
            agents[id].cumulative_potential = 0.0f;
        }
        else if (sensed_potential - agents[id].previous_potential < -ODOR_THRESHOLD){ //|| auto_transition_probability<AUTO_TRANSITION_PROBABILITY_THRESHOLD){ //moving in the wrong direction, then PIROUETTE
            agents[id].state = 1;
            agents[id].cumulative_potential += (sensed_potential - agents[id].previous_potential);
        }

        float fx, fy, new_angle;
        float mu, kappa, scale, shape;
        int sub_state = agents[id].substate;
        int base_index = id * N_STATES;
        ExplorationState* explorationState = &d_explorationStates[base_index + sub_state];
        float* probabilities = explorationState->probabilities;
        mu = explorationState->angle_mu;    // this can be negative, set below, 50% chance
        kappa = explorationState->angle_kappa;
        scale = explorationState->speed_scale;
        shape = explorationState->speed_spread;
        float random_angle = (float)explorationState->angle_mu_sign * sample_from_von_mises(mu, kappa, states);//wrapped_cauchy(0.0, 0.6, &states[id]);//curand_normal(&states[id]) * M_PI/4;////

        float lambda=0.0f; //@TODO: try to make it a function of the potential (and re-add the potential)

        if(agents[id].state == 0){ //if the agent is moving = RUN - LOW TURNING - EXPLOIT
            //if the max concentration is 0 (or best direction is 0,0), then choose random only (atan will give unreliable results)
            if (max_concentration< ODOR_THRESHOLD || (max_concentration_x==0 && max_concentration_y==0) ) {

                agents[id].angle += ((1.0f-lambda)* random_angle);
            }
            else {
                float norm = sqrt(max_concentration_x * max_concentration_x + max_concentration_y * max_concentration_y);
                float direction_x = max_concentration_x / norm;
                float direction_y = max_concentration_y / norm;
                float bias = atan2(direction_y, direction_x);

                float current_angle = agents[id].angle;
                if(bias-current_angle>=0){
                    bias = M_PI / 4;
                } else{
                    bias = -M_PI / 4;
                }

                float k = KAPPA;// * pow(sensed_potential / max_concentration, 2);
                new_angle = sample_from_von_mises(bias, k, &states[id]);

                agents[id].angle += new_angle;
            }

        }
        else{ //BROWNIAN MOTION - HIGH TURNING - EXPLORE
            agents[id].angle += random_angle;

        }

        if(agents[id].angle>2 * M_PI || agents[id].angle<-2 * M_PI){
            agents[id].angle = fmodf(agents[id].angle, 2*M_PI);
        }

        fx = cosf(agents[id].angle);
        fy = sinf(agents[id].angle);

        float new_speed = curand_log_normal(&states[id], logf(scale), shape);
        while(new_speed>MAX_ALLOWED_SPEED) new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //printf("New Speed: %f with scale %f and shape %f\n", new_speed, scale, shape);
        if (sub_state==5){ //reversals
            new_speed = - new_speed;
        }
        float dx = fx * new_speed;
        float dy = fy * new_speed;

        agents[id].previous_potential = sensed_potential;
        agents[id].x += dx;
        agents[id].y += dy;
        agents[id].speed = new_speed;
        // Apply periodic boundary conditions
        if (agents[id].x < 0) agents[id].x += WIDTH;
        if (agents[id].x >= WIDTH) agents[id].x -= WIDTH;
        if (agents[id].y < 0) agents[id].y += HEIGHT;
        if (agents[id].y >= HEIGHT) agents[id].y -= HEIGHT;
        int new_x = (int)(agents[id].x / DX);
        int new_y = (int)(agents[id].y / DY);


        //printf("Current substate: %d, max duration %d id again (just to be sure) %d\n", agents[id].substate, explorationState->max_duration, explorationState->id);

        if(explorationState->duration>0){
            explorationState->duration--;
        }
        if(explorationState->duration<=0){
            agents[id].substate = select_next_state(probabilities, &states[id], N_STATES);
            //printf("switching to state %d\n", agents[id].substate);
            explorationState = &d_explorationStates[base_index + agents[id].substate];
            if(agents[id].substate<3) {
                if(explorationState->max_duration>=0) {
                    int duration = (int) curand_log_normal(&states[id], explorationState->duration_mu,
                                                           explorationState->duration_sigma);
                    //printf("New Duration: %d mean %f std %f max bound %d\n", duration, explorationState->duration_mu,
                           //explorationState->duration_sigma, explorationState->max_duration);
                    while (duration <= 0 || duration > explorationState->max_duration) {
                        duration = (int) curand_log_normal(&states[id], explorationState->duration_mu,
                                                           explorationState->duration_sigma);
                        //printf("Refusing New Duration %d with Bound %d\n", duration, explorationState->max_duration);
                    }
                    explorationState->duration = duration;
                }
                else{
                    explorationState->duration = 0;
                    printf("Duration lower than 0\n");
                }
            }
            else{
                explorationState->duration = 0;
            }

        }
        //IF the new substate is different from the previous one, then choose sign for the angle mu and augments timesteps in this substate, otherwise set to 0
        if(agents[id].substate != agents[id].previous_substate){
            if(curand_uniform(&states[id])>0.5){
                explorationState->angle_mu_sign *= -1;
            }
            explorationState->timesteps_in_state++;
        } else {
            explorationState->timesteps_in_state = 0;
        }

        agents[id].previous_substate = sub_state;
        //check if the agent is in the target area
        if (new_x >= 3* N/4 - TARGET_AREA_SIDE_LENGTH/2 && new_x < 3*N/4 + TARGET_AREA_SIDE_LENGTH/2 && new_y >= N/2 - TARGET_AREA_SIDE_LENGTH/2 && new_y < N/2 + TARGET_AREA_SIDE_LENGTH/2){
            agents[id].is_agent_in_target_area = 1;
            agents[id].steps_in_target_area++;
            if(agents[id].first_timestep_in_target_area == -1){
                agents[id].first_timestep_in_target_area = timestep;
            }
        }
        else{
            agents[id].is_agent_in_target_area = 0;
        }

    }
}
#endif //UNTITLED_AGENT_UPDATE_H
