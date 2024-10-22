//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_AGENT_UPDATE_H
#define UNTITLED_AGENT_UPDATE_H
#include <cuda_runtime.h>
#include <random>
#include <limits>
#include <cmath>
// Function to sample from a von Mises distribution
__device__ float sample_from_von_mises(float mu, float kappa, curandState* state) {
    // Handle kappa = 0 (uniform distribution)
    if (kappa < std::numeric_limits<float>::epsilon()) {
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
__global__ void moveAgents(Agent* agents, curandState* states, float* potential, int* agent_count_grid, int worm_count, int timestep, float sigma) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {

        int max_concentration_x = 0;
        int max_concentration_y = 0;
        int agent_x = (int) (agents[id].x / DX);
        int agent_y = (int) (agents[id].y / DY);
        float sensed_potential = potential[agent_x * N + agent_y];
        //add a small perceptual noise to the potential
        if(sigma!=0.0f){
            float perceptual_noise = curand_normal(&states[id]) * sigma;
            if(perceptual_noise>sigma) perceptual_noise = sigma;
            if(perceptual_noise<-sigma) perceptual_noise = (-sigma);
            sensed_potential += perceptual_noise;
        }
        float max_concentration = sensed_potential;
        //printf("Sensed potential: %f\n", sensed_potential);
        for (int i = -SENSING_RANGE; i <= SENSING_RANGE; ++i) {
            for (int j = -SENSING_RANGE; j <= SENSING_RANGE; ++j) {
                float concentration = 0.0f;
                int xIndex = agent_x + i;
                int yIndex = agent_y + j;
                //apply periodic boundary conditions
                if (xIndex < 0) xIndex += N;
                if (xIndex >= N) xIndex -= N;
                if (yIndex < 0) yIndex += N;
                if (yIndex >= N) yIndex -= N;
                if (xIndex >= 0 && xIndex < N && yIndex >= 0 && yIndex < N) {
                    concentration = potential[xIndex * N + yIndex];
                    //printf("At (%d, %d) concentration: %f\n", xIndex, yIndex, concentration);
                    /*if (sigma != 0.0f) {
                        float perceptual_noise = curand_normal(&states[id]) * sigma;
                        if (perceptual_noise > sigma) perceptual_noise = sigma;
                        if (perceptual_noise < -sigma) perceptual_noise = (-sigma);
                        concentration += perceptual_noise;
                    }*/
                    if (concentration > max_concentration) {
                        max_concentration = concentration;
                        max_concentration_x = i;
                        max_concentration_y = j;
                        //printf("Max concentration: %f in direction %d,%d\n", max_concentration, max_concentration_x, max_concentration_y);
                    }
                    //printf("At (%d, %d) concentration: %f\n", xIndex, yIndex, concentration);
                }

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

        //printf("Agent %d state: %d\n", id, agents[id].state);
        float fx, fy, new_angle, new_direction_x, new_direction_y;

        float random_angle = curand_normal(&states[id]) * M_PI/4;
        while(random_angle > M_PI || random_angle < -M_PI){
            random_angle = curand_normal(&states[id]) * M_PI/4;
        }
        float lambda=0.0f;
        float random_brownian_angle = (2.0f * curand_uniform(&states[id]) - 1.0f) * M_PI;
        // no need to check if the angle is between -pi and pi; done in the python script

        if(agents[id].state == 0){ //if the agent is moving = RUN - LOW TURNING - LEVY FLIGHT
            //if the max concentration is 0 (or best direction is 0,0), then choose random only (atan will give unreliable results)
            if (max_concentration< ODOR_THRESHOLD || (max_concentration_x==0 && max_concentration_y==0) ) {

                agents[id].angle += ((1.0f-lambda)* random_angle);
            }
            else {
                //max_concentration_x = -max_concentration_x;
                //max_concentration_y = -max_concentration_y;
                float norm = sqrt(max_concentration_x * max_concentration_x + max_concentration_y * max_concentration_y);
                float direction_x = max_concentration_x / norm;
                float direction_y = max_concentration_y / norm;
                //printf("Max concentration: %f in direction %f,%f\n", max_concentration, direction_x, direction_y);
                float bias = atan2(direction_y, direction_x);

                //printf("Bias: %f\n", bias);
                float current_angle = agents[id].angle;
                if(bias-current_angle>0){
                    bias = M_PI / 4;
                } else if(bias-current_angle<0){
                    bias = -M_PI / 4;
                } else{
                    bias = 0.0f;
                }
                //make bias modulo pi/2
                //bias = fmodf(bias, 2*M_PI);
                float k = KAPPA;// * pow(sensed_potential / max_concentration, 2);
                new_angle = sample_from_von_mises(bias, k, &states[id]);
                //printf("New angle: %f\n", new_angle);
                //new_direction_x = cosf(new_angle);
                //new_direction_y = sinf(new_angle);
                //lambda = pow(sensed_potential / max_concentration, 2);
                //lambda = 0.0f;
                //lambda = ((agents[id].previous_potential - sensed_potential) / max_concentration);
                //lambda = 1.0f-abs(new_angle)/(M_PI);
                //lambda=0.0f;
                //printf("Lambda: %f\n", lambda);
                agents[id].angle += new_angle;
            }

        }
        else{ //BROWNIAN MOTION - HIGH TURNING - PIROUETTE
            agents[id].angle += random_angle;

        }

        if(agents[id].angle>2 * M_PI || agents[id].angle<-2 * M_PI){
            //printf("Angle: %f PORCODIOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n", agents[id].angle);
            agents[id].angle = fmodf(agents[id].angle, 2*M_PI);
        }

        fx = cosf(agents[id].angle);
        fy = sinf(agents[id].angle);
        //agents[id].angle = fmodf(agents[id].angle, 2.0f * M_PI);

        float new_speed = SPEED;
        if(max_concentration>ODOR_THRESHOLD){ //on food - lognorm distribution of speed
            float scale, shape;
            if(curand_uniform(&states[id])<ON_FOOD_SPEED_SLOW_WEIGHT){
                scale = ON_FOOD_SPEED_SCALE_SLOW;
                shape = ON_FOOD_SPEED_SHAPE_SLOW;
            }
            else{
                scale = ON_FOOD_SPEED_SCALE_FAST;
                shape = ON_FOOD_SPEED_SHAPE_FAST;
            }
            new_speed = curand_log_normal(&states[id], logf(scale), shape);
            while(new_speed>MAX_ALLOWED_SPEED) new_speed = curand_log_normal(&states[id], logf(scale), shape);

        }
        else{ //off food - lognormal distribution of speed
            float scale, shape;
            if(curand_uniform(&states[id])<OFF_FOOD_SPEED_SLOW_WEIGHT){
                scale = OFF_FOOD_SPEED_SCALE_SLOW;
                shape = OFF_FOOD_SPEED_SHAPE_SLOW;
            }
            else{
                scale = OFF_FOOD_SPEED_SCALE_FAST;
                shape = OFF_FOOD_SPEED_SHAPE_FAST;
            }
            new_speed = curand_log_normal(&states[id], logf(scale), shape);
            while(new_speed>MAX_ALLOWED_SPEED) new_speed = curand_log_normal(&states[id], logf(scale), shape);

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

        if(ENABLE_MAXIMUM_NUMBER_OF_AGENTS_PER_CELL){
            // Check if the new cell is full
            if (agent_count_grid[new_x * N + new_y] >= MAXIMUM_AGENTS_PER_CELL) {
                // Create an array of indices representing the neighboring cells
                int indices[] = {0, 1, 2, 3};
                // Shuffle the array of indices
                for (int k = 3; k > 0; --k) {
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

        if (agent_x != new_x || agent_y != new_y) {

            // Decrease the count in the old cell
            atomicAdd(&agent_count_grid[agent_x * N + agent_y], -1);

            // Increase the count in the new cell
            atomicAdd(&agent_count_grid[new_x * N + new_y], 1);

        }

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
