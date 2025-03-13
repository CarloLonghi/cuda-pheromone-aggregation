//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_AGENT_UPDATE_H
#define UNTITLED_AGENT_UPDATE_H
#include <cuda_runtime.h>
#include <random>
#include <limits>
#include <cmath>
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
__global__ void moveAgents(Agent* agents, curandState* states,  float* potential, /*int* agent_count_grid,*/ int worm_count, int timestep,
    float sigma, float k) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {

        float max_concentration_x = 0.0;
        float max_concentration_y = 0.0;
        int agent_x = (int)round(agents[id].x / DX), agent_y = (int)round(agents[id].y / DX);
        float sensed_potential = potential[agent_x * N + agent_y];//potential[agent_x * N + agent_y];
        //sensed_potential = ATTRACTION_STRENGTH * logf(sensed_potential + ATTRACTION_SCALE);
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
            int sample_x = (int)round((agents[id].x + SENSING_RADIUS * cosf(angle))/DX);
            int sample_y = (int)round((agents[id].y + SENSING_RADIUS * sinf(angle))/DY);
            float concentration = potential[sample_x * N + sample_y];
            // Add perceptual noise if sigma is not zero
            if (sigma != 0.0f) {
                concentration += curand_normal(&states[id]) * sigma;
            }
            //concentration = ATTRACTION_STRENGTH * logf(concentration + ATTRACTION_SCALE);

            if (abs(concentration) > abs(max_concentration)) {
                max_concentration = concentration;
                max_concentration_x = cosf(angle);
                max_concentration_y = sinf(angle);
            }
        }

        float fx, fy, new_angle;
        // float mu, kappa;
        // mu = curand_uniform(&states[id]) * 2 * M_PI;
        // kappa = 7;
        // scale = explorationState->speed_scale;
        // shape = explorationState->speed_spread;
        //float random_angle = curand_normal(&states[id]) * M_PI/4;//sample_from_von_mises(mu, kappa, &states[id]);//wrapped_cauchy(0.0, 0.6, &states[id]);////

        float random_angle = sample_from_von_mises(agents[id].angle, k, &states[id]);
        new_angle = random_angle;

        if (abs(max_concentration)>=PHEROMONE_THRESHOLD && (max_concentration_x!=0 && max_concentration_y!=0) ) {
            // Brownian Motion
            float norm = sqrt(max_concentration_x * max_concentration_x + max_concentration_y * max_concentration_y);
            float direction_x = max_concentration_x / norm;
            float direction_y = max_concentration_y / norm;
            float bias = atan2(direction_y, direction_x);

            if(bias-new_angle>=0){
                if(bias-new_angle>=M_PI){
                    bias = -M_PI / 4;
                }
                else{
                    bias = M_PI / 4;
                }
            }else{
                if(bias-new_angle<-M_PI){
                    bias = M_PI / 4;
                }

                else{
                    bias = -M_PI / 4;
                }
            }
            new_angle += bias;
        }

        agents[id].angle = new_angle;

        if(agents[id].angle>2 * M_PI || agents[id].angle<-2 * M_PI){
            agents[id].angle = fmodf(agents[id].angle, 2*M_PI);
        }

        fx = cosf(agents[id].angle);
        fy = sinf(agents[id].angle);

        float norm = sqrt(fx * fx + fy * fy);
        fx = fx / norm;
        fy = fy / norm;

        float new_speed = SPEED;
        //float new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //while(new_speed>MAX_ALLOWED_SPEED) new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //printf("New Speed: %f with scale %f and shape %f\n", new_speed, scale, shape);
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
    }
}
#endif //UNTITLED_AGENT_UPDATE_H
