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

        int agent_x = (int)round(agents[id].x / DX), agent_y = (int)round(agents[id].y / DY);
        float sensed_potential = potential[agent_x * N + agent_y] + curand_normal(&states[id]) * SENSING_NOISE;//potential[agent_x * N + agent_y];
        //sensed_potential = ATTRACTION_STRENGTH * logf(sensed_potential + ATTRACTION_SCALE);
        //add a small perceptual noise to the potential
        if(sigma!=0.0f){
            float perceptual_noise = curand_normal(&states[id]) * sigma;
            if(perceptual_noise>sigma) perceptual_noise = sigma;
            if(perceptual_noise<-sigma) perceptual_noise = (-sigma);
            sensed_potential += perceptual_noise;
        }

        // compute tumble rate
        int tail_x = (int)round((agents[id].x - 0.1 * cosf(agents[id].angle)) / DX);
        int tail_y = (int)round((agents[id].y - 0.1 * sinf(agents[id].angle)) / DY);
        float tail_potential = potential[tail_x * N + tail_y] + curand_normal(&states[id]) * SENSING_NOISE;
        // float dp = sensed_potential - agents[id].previous_potential;
        float dp = sensed_potential - tail_potential;
        float r = 1 / (1 + exp(100 * (dp + k)));
        // r = 0.032256911591854065; // to reproduce videos of N2 diffusion

        float fx, fy;
        float p = curand_uniform(&states[id]);
        if (p < r){
            float random_angle = curand_uniform(&states[id]) * M_PI * 2;
            agents[id].angle = random_angle;
        }

        if(agents[id].angle>2 * M_PI || agents[id].angle<-2 * M_PI){
            agents[id].angle = fmodf(agents[id].angle, 2*M_PI);
        }

        fx = cosf(agents[id].angle);
        fy = sinf(agents[id].angle);

        // find neighbors alignment angle
        int num_neighbors = 0;
        float align_x = 0, align_y = 0, anglediff;
        for (int i = 0; i < WORM_COUNT; ++i){
            if (i != id){
                float diffx = agents[id].x - agents[i].x;
                float diffy = agents[id].y - agents[i].y;
                float dist = sqrt(diffx * diffx + diffy * diffy);
                if (dist < ALIGNMENT_RADIUS){
                    num_neighbors += 1;
                    anglediff = agents[id].angle - agents[i].angle;
                    if (anglediff > M_PI){
                        anglediff = 2 * M_PI - anglediff;
                    }
                    else if (anglediff < - M_PI)
                    {
                        anglediff = 2 * M_PI + anglediff;
                    }
                    align_x += cosf(anglediff);
                    align_y += sinf(anglediff);
                }
            }
        }
        align_x /= num_neighbors;
        align_y /= num_neighbors;

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

        if (num_neighbors > 0){
            fx += align_x * ALIGNMENT_STRENGTH;
            fy += align_y * ALIGNMENT_STRENGTH;
        }

        float norm = sqrt(fx * fx + fy * fy);
        fx = fx / norm;
        fy = fy / norm;

        float new_speed = SPEED;
        //float new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //while(new_speed>MAX_ALLOWED_SPEED) new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //printf("New Speed: %f with scale %f and shape %f\n", new_speed, scale, shape);
        float dx = fx * new_speed;
        float dy = fy * new_speed;

        // apply boundary conditions
        if (dx + agents[id].x >= WIDTH){
            dx = WIDTH - agents[id].x;
            if (dy >= 0) dy = sqrt(new_speed * new_speed - dx * dx);
            else dy = - sqrt(new_speed * new_speed - dx * dx);
            
        }
        else if (dx + agents[id].x < 0){
            dx = -agents[id].x;
            if (dy >= 0) dy = sqrt(new_speed * new_speed - dx * dx);
            else dy = - sqrt(new_speed * new_speed - dx * dx);            
        }

        if (dy + agents[id].y >= HEIGHT){
            dy = HEIGHT - agents[id].y;
            if (dx >= 0) dx = sqrt(new_speed * new_speed - dy * dy);
            else dx = - sqrt(new_speed * new_speed - dy * dy);
        }
        else if (dy + agents[id].y < 0){
            dy = -agents[id].y;
            if (dx >= 0) dx = sqrt(new_speed * new_speed - dy * dy);
            else dx = - sqrt(new_speed * new_speed - dy * dy);            
        }

        agents[id].previous_potential = sensed_potential;
        agents[id].x += dx;
        agents[id].y += dy;
        agents[id].speed = new_speed;


    }
}
#endif //UNTITLED_AGENT_UPDATE_H
