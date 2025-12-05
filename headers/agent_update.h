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

__device__ float gaussian_core_force(float r, float epsr, float epsa) {
    float sr = 1.0;
    float sa = 2.0;
    return 2 * r / (sr * sr) * epsr * std::exp(-(r/sr)*(r/sr)) - 2 * r / (sa * sa) * epsa * std::exp(-(r/sa)*(r/sa));
}

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


// CUDA kernel to update the position of each agent
__global__ void moveAgents(Agent* agents, curandState* states, float* potential,
     /*int* agent_count_grid,*/ int worm_count, int timestep, float sigma, float align_strength, float slow_factor,
    float espa, float espr) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {

        int num_neighbors = 0;
        float angle_x = 0, angle_y = 0;
        float attr_x = 0, attr_y = 0, rep_x = 0, rep_y = 0, angle_diff = 0, align_x = 0, align_y = 0, align_angle = 0, total_inf = 0, agr = 0; 

        for (int j = 0; j < worm_count; j++){
            if (j != id) {  // Skip self-interaction
                float diffx = agents[id].x - agents[j].x;
                float diffy = agents[id].y - agents[j].y;
                float dist = sqrt(diffx * diffx + diffy * diffy);
                if (dist < ALIGNMENT_RADIUS){
                    num_neighbors += 1;
                    align_x += cosf(agents[j].angle - agents[id].angle) * cosf(agents[j].angle);
                    align_y += cosf(agents[j].angle - agents[id].angle) * sinf(agents[j].angle);                      

                    total_inf += (1 - dist / ALIGNMENT_RADIUS); 
                    
                    if (dist > 0){
                        float gcf = gaussian_core_force(dist / ALIGNMENT_RADIUS * 4, espr, espa);
                        attr_x += gcf * (agents[id].x - agents[j].x) / dist;
                        attr_y += gcf * (agents[id].y - agents[j].y) / dist;
                    }         
                }
            }
        }

        float fx, fy, nx, ny;

        if (num_neighbors > 0){
            float norm = sqrt(align_x * align_x + align_y * align_y);
            agr = norm / num_neighbors;
            align_x /= norm;
            align_y /= norm;
            float align_angle = atan2(align_y, align_x);
            float diff = align_angle - agents[id].angle;
            diff = atan2(sinf(diff), cosf(diff));
            agents[id].angle += (align_strength * diff) * DT;
        }

        float theta = 100;
        float sigma = 0.06;
        agents[id].omega += (-(agents[id].omega / theta) + curand_normal(&states[id]) * (sigma * sigma)) * DT;
        agents[id].angle += (agents[id].omega) * DT;

        fx = cosf(agents[id].angle) + attr_x + rep_x;
        fy = sinf(agents[id].angle) + attr_y + rep_y;
        float norm = sqrt(fx * fx + fy * fy);
        fx /= norm;
        fy /= norm;
        agents[id].angle = atan2(fy, fx);

        int agent_x = (int)round(agents[id].x / DX), agent_y = (int)round(agents[id].y / DY);
        float sensed_potential = potential[agent_x * NN+ agent_y];//potential[agent_x *+ agent_y];
        //sensed_potential = ATTRACTION_STRENGTH * logf(sensed_potential + ATTRACTION_SCALE);
        //add a small perceptual noise to the potential

        // compute tumble rate
        int tail_x = (int)round((agents[id].x - BODY_LENGTH * cosf(agents[id].angle)) / DX);
        int tail_y = (int)round((agents[id].y - BODY_LENGTH * sinf(agents[id].angle)) / DY);
        float tail_potential = potential[tail_x * NN + tail_y];
        float dp = sensed_potential - tail_potential;
        float r = DT * ((1 / (1 + expf(dp * 100 + 1.359744321607823))) * 0.06 + 0.02);
        r = DT * 0.032256911591854065 / 4;
        r = DT * 0.008064227897963516;

        float p = curand_uniform(&states[id]);
        if (p < r){
            float random_angle = curand_uniform(&states[id]) * M_PI * 2;
            agents[id].angle = random_angle;
            fx = cosf(agents[id].angle);
            fy = sinf(agents[id].angle);                
        }       

        float new_speed = SPEED;
        if (num_neighbors > 0 & slow_factor > 0){
            new_speed *= exp(slow_factor * -total_inf);
        }
        new_speed += curand_normal(&states[id]) * GAMMA;

        //float new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //while(new_speed>MAX_ALLOWED_SPEED) new_speed = curand_log_normal(&states[id], logf(scale), shape);
        //printf("New Speed: %f with scale %f and shape %f\n", new_speed, scale, shape);
        float dx = fx * new_speed;
        float dy = fy * new_speed;

        // apply boundary conditions
        if (dx + agents[id].x >= WIDTH){
            dx = WIDTH - agents[id].x;
            if (dy >= 0) dy = min(sqrt(new_speed * new_speed - dx * dx), HEIGHT - agents[id].y);
            else dy = - min(sqrt(new_speed * new_speed - dx * dx), agents[id].y);
            
        }
        else if (dx + agents[id].x < 0){
            dx = -agents[id].x;
            if (dy >= 0) dy = min(sqrt(new_speed * new_speed - dx * dx), HEIGHT - agents[id].y);
            else dy = - min(sqrt(new_speed * new_speed - dx * dx), agents[id].y);            
        }

        if (dy + agents[id].y >= HEIGHT){
            dy = HEIGHT - agents[id].y;
            if (dx >= 0) dx = min(sqrt(new_speed * new_speed - dy * dy), WIDTH - agents[id].x);
            else dx = - min(sqrt(new_speed * new_speed - dy * dy), agents[id].x);
        }
        else if (dy + agents[id].y < 0){
            dy = -agents[id].y;
            if (dx >= 0) dx = min(sqrt(new_speed * new_speed - dy * dy), WIDTH - agents[id].x);
            else dx = - min(sqrt(new_speed * new_speed - dy * dy), agents[id].x);            
        }           
        
        agents[id].previous_potential = sensed_potential;
        agents[id].x += dx;
        agents[id].y += dy;
        agents[id].speed = new_speed;

        // if (agents[id].x < 0) agents[id].x += WIDTH;
        // if (agents[id].x >= WIDTH) agents[id].x -= WIDTH;
        // if (agents[id].y < 0) agents[id].y += HEIGHT;
        // if (agents[id].y >= HEIGHT) agents[id].y -= HEIGHT;
    }
}
#endif //UNTITLED_AGENT_UPDATE_H
