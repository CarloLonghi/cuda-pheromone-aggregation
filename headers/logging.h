//
// Created by nema on 03/10/24.
//

#ifndef UNTITLED_LOGGING_H
#define UNTITLED_LOGGING_H
#include <cuda_runtime.h>

using json = nlohmann::json;

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

void savePositionsToJSON(const char* filename, float* positions, int worm_count, int n_steps, bool one_parameter=false){
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
            if(!one_parameter) {
                log[std::to_string(i)].push_back(
                        {positions[(j * worm_count + i) * 2], positions[(j * worm_count + i) * 2 + 1]});
            }
            else {
                log[std::to_string(i)].push_back(
                        {positions[j * worm_count + i]});
            }
        }
    }

    std::ofstream outFile(filename);
    outFile << log.dump(4);  // Pretty-print JSON with an indentation of 4 spaces
    outFile.close();
}

void saveInsideAreaToJSON(const char* filename, Agent* h_agents, int worm_count, int n_steps) {
    json log;

    // Log simulation parameters
    log["parameters"] = {{"WIDTH",            WIDTH},
                         {"HEIGHT",           HEIGHT},
                         {"N", worm_count},
                         {"LOGGING_INTERVAL", LOGGING_INTERVAL},
                         {"N_STEPS",          N_STEPS}};

    std::ofstream outFile(filename);
    outFile << log.dump();  // Pretty-print JSON with an indentation of 4 spaces
    outFile.close();
}

void saveAllDataToJSON(const char* filename, float* positions, Agent* agents, int worm_count, int n_steps) {
    nlohmann::json json_data;
    if(LOG_POSITIONS){
    json_data["positions"] = nlohmann::json::array();
    }
    json_data["inside_area"] = nlohmann::json::array();
    json_data["parameters"] = {{"WIDTH",            WIDTH},
                               {"HEIGHT",           HEIGHT},
                               {"N",                N},
                               {"WORM_COUNT",       WORM_COUNT},
                               {"LOGGING_INTERVAL", LOGGING_INTERVAL},
                               {"TIME",          TIME},
                               {"SENSING_RADIUS",   SENSING_RADIUS},
                               {"SPEED",            SPEED},
                               {"PHEROMONE_THRESHOLD",   PHEROMONE_THRESHOLD},
                               {"MAX_CONCENTRATION",    MAX_CONCENTRATION}};
    for (int i = 0; i < worm_count; ++i) {
        nlohmann::json agent_data;
        if(LOG_POSITIONS){
        agent_data["positions"] = nlohmann::json::array();
        }
        if(LOG_POSITIONS) {
            for (int j = 0; j < n_steps; ++j) {
                agent_data["positions"].push_back(
                    {positions[(j * worm_count + i) * 2], positions[(j * worm_count + i) * 2 + 1]});
            }
        }
        if(LOG_POSITIONS){
        json_data["positions"].push_back(agent_data["positions"]);
        }
    }

    std::ofstream file(filename);
    file << json_data.dump(4);
    file.close();
}

#endif //UNTITLED_LOGGING_H
