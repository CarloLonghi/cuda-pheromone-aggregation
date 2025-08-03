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

__device__ int2 getGridCell(float2 pos) {
    return make_int2(floor(pos.x / CELL_SIZE),
                    floor(pos.y / CELL_SIZE));
}

__device__ unsigned int flattenCellIndex(int2 cell) {
    return cell.x * GRID_DIM_X + cell.y;
}

// CUDA kernel to update the position of each agent
__global__ void moveAgents(Agent* agents, curandState* states, int* cellStart, int* cellEnd,  float* potential,
     /*int* agent_count_grid,*/ int worm_count, int timestep, float sigma, float align_strength, float slow_factor, int slow_nc) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {

        int agent_x = (int)round(agents[id].x / DX), agent_y = (int)round(agents[id].y / DY);
        float sensed_potential = potential[agent_x * NN+ agent_y] + curand_normal(&states[id]) * SENSING_NOISE;//potential[agent_x *+ agent_y];
        //sensed_potential = ATTRACTION_STRENGTH * logf(sensed_potential + ATTRACTION_SCALE);
        //add a small perceptual noise to the potential
        if(sigma!=0.0f){
            float perceptual_noise = curand_normal(&states[id]) * sigma;
            if(perceptual_noise>sigma) perceptual_noise = sigma;
            if(perceptual_noise<-sigma) perceptual_noise = (-sigma);
            sensed_potential += perceptual_noise;
        }

        // compute tumble rate
        int tail_x = (int)round((agents[id].x - BODY_LENGTH * cosf(agents[id].angle)) / DX);
        int tail_y = (int)round((agents[id].y - BODY_LENGTH * sinf(agents[id].angle)) / DY);
        float tail_potential = potential[tail_x * NN + tail_y] + curand_normal(&states[id]) * SENSING_NOISE;
        float dp = sensed_potential - tail_potential;
        float r = (1 / (1 + expf(dp * 100 + 0.3699060651715053))) * 0.006 + 0.002;
        // float r = 0.0032256911591854065; // to reproduce videos of N2 diffusion

        int num_neighbors = 0, na = 0;
        float angle_x = 0, angle_y = 0;
        float attr_x = 0, attr_y = 0, rep_x = 0, rep_y = 0, angle_diff = 0, align_x = 0, align_y = 0; 

        int2 cell = getGridCell(make_float2(agents[id].x, agents[id].y));
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                int2 neighborCell = make_int2(cell.x + x, cell.y + y);
                
                // Skip if cell is outside simulation bounds
                if (neighborCell.x >= 0 & neighborCell.y >= 0 &
                    neighborCell.x < GRID_DIM_X & neighborCell.y < GRID_DIM_Y){

                    unsigned int neighborHash = flattenCellIndex(neighborCell);
                    
                    // Get start and end index for this cell
                    int startIdx = cellStart[neighborHash];
                    int endIdx = cellEnd[neighborHash];                   

                    // Iterate through all particles in this cell
                    for(int j = startIdx; j < endIdx; j++) {
                        if (j != id) {  // Skip self-interaction
                            float diffx = agents[id].x - agents[j].x;
                            float diffy = agents[id].y - agents[j].y;
                            float dist = sqrt(diffx * diffx + diffy * diffy);
                            if (dist < ALIGNMENT_RADIUS){
                                num_neighbors += 1;
                                if (sin(agents[j].angle) > 0){
                                    angle_x += cos(agents[j].angle);
                                    angle_y += sin(agents[j].angle);
                                }
                                else{
                                    angle_x += -cos(agents[j].angle);
                                    angle_y += -sin(agents[j].angle);
                                }

                                if (dist < REPULSION_RADIUS){
                                    rep_x += 0.1 * (dist - REPULSION_RADIUS) * (agents[j].x - agents[id].x) / dist;
                                    rep_y += 0.1 * (dist - REPULSION_RADIUS) * (agents[j].y - agents[id].y) / dist;
                                }
                                if (REPULSION_RADIUS <= dist & dist < ALIGNMENT_RADIUS){
                                    attr_x += (0.00002 / dist) * (agents[j].x - agents[id].x) / dist;
                                    attr_y += (0.00002 / dist) * (agents[j].y - agents[id].y) / dist;

                                    na += 1;
                                    angle_diff = agents[j].angle - agents[id].angle;
                                    if (angle_diff > (M_PI / 2) & angle_diff < M_PI) {
                                        align_x += cos(agents[j].angle - M_PI);
                                        align_y += sin(agents[j].angle - M_PI);
                                    }
                                    else if (angle_diff < (-M_PI / 2) & angle_diff > (-M_PI)) {
                                        align_x += cos(agents[j].angle + M_PI);
                                        align_y += sin(agents[j].angle + M_PI);
                                    }
                                    else {
                                        align_x += cos(agents[j].angle);
                                        align_y += sin(agents[j].angle);
                                    }                        
                                }
                            }
                        }
                    }
                }
            }
        }
        if (na > 0){
            align_x /= na;
            align_y /= na;
        }

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

        float sum_length = sqrt(angle_x*angle_x + angle_y*angle_y) / num_neighbors;

        if (num_neighbors > 0){
            fx += align_x * (align_strength * sum_length) + attr_x + rep_x;
            fy += align_y * (align_strength * sum_length) + attr_y + rep_y;    
        }        

        float norm = sqrt(fx * fx + fy * fy);
        fx = fx / norm;
        fy = fy / norm;

        float new_speed = SPEED;
        if (num_neighbors > 0){
            new_speed /= 1 + (slow_factor  * (1 - sum_length) * (min(slow_nc, num_neighbors) / slow_nc));
        }

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

        agents[id].angle = atan2(fy, fx);
        if(agents[id].angle>2 * M_PI || agents[id].angle<-2 * M_PI){
            agents[id].angle = fmodf(agents[id].angle, 2*M_PI);
        }        

        agents[id].previous_potential = sensed_potential;
        agents[id].x += dx;
        agents[id].y += dy;
        agents[id].speed = new_speed;


    }
}
#endif //UNTITLED_AGENT_UPDATE_H
