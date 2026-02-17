#include <fstream>
#include <vector>
#include <iostream>
#include <vector>
#include "../Eigen/Dense"
#include <stdio.h>
#include <zlib.h>

using namespace Eigen;

#define TIME 300
#define DT 0.1f
#define N_STEPS int(TIME / DT)
#define LOGGING_INTERVAL 10
#define WORM_COUNT 10000
#define WIDTH 128
#define HEIGHT 128

// Agent parameters
#define BODY_LENGTH 0.25f
#define SPEED 0.4f * DT // (0.3f * BODY_LENGTH * DT)

// Descriptor parameters
#define CLUSTERING_RADIUS 0.5
#define NEIGHBOR_RADIUS 2.0
#define MSD_WINDOW 1

void saveCompressed(const float* data, size_t size, const std::string& filename) {
    uLongf compressed_size = compressBound(size * sizeof(float));
    std::vector<Bytef> compressed(compressed_size);
    
    compress(compressed.data(), &compressed_size,
             reinterpret_cast<const Bytef*>(data),
             size * sizeof(float));
    
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(compressed.data()), compressed_size);
}

float* loadCompressed(const std::string& filename, size_t original_size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    size_t compressed_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<Bytef> compressed(compressed_size);
    file.read(reinterpret_cast<char*>(compressed.data()), compressed_size);
    
    float* data = new float[original_size];

    uLongf decompressed_size = original_size * sizeof(float);
    uncompress(reinterpret_cast<Bytef*>(data), &decompressed_size,
               compressed.data(), compressed_size);

    return data;
}

Matrix2d computeRegularizedCovariance(const std::vector<Vector2d>& points, double epsilon = 1e-6) {
    if (points.empty()) return Matrix2d::Zero();
    
    // Compute mean
    Vector2d mean = Vector2d::Zero();
    for (const auto& p : points) {
        mean += p;
    }
    mean /= points.size();
    
    // Compute covariance
    Matrix2d cov = Matrix2d::Zero();
    for (const auto& p : points) {
        Vector2d diff = p - mean;
        cov += diff * diff.transpose();
    }
    cov /= (points.size() - 1);
    
    // Add regularization
    cov += epsilon * Matrix2d::Identity();
    
    return cov;
}

std::vector<Vector2d> convert1DArrayTo2DVector(std::vector<float> data, int num_points) {
    std::vector<Vector2d> points;
    points.reserve(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        double x = data[2 * i];      // x coordinate
        double y = data[2 * i + 1];  // y coordinate
        points.emplace_back(x, y);
    }
    
    return points;
}

Matrix2d computeCovariance(const std::vector<Vector2d>& points) {
    if (points.empty()) return Matrix2d::Zero();
    
    // Compute mean
    Vector2d mean = Vector2d::Zero();
    for (const auto& p : points) {
        mean += p;
    }
    mean /= points.size();
    
    // Compute covariance
    Matrix2d cov = Matrix2d::Zero();
    for (const auto& p : points) {
        Vector2d diff = p - mean;
        cov += diff * diff.transpose();
    }
    cov /= (points.size() - 1);  // Sample covariance
    
    return cov;
}

// Function to perform DFS traversal
void dfs(bool *adjMatrix, bool visited[WORM_COUNT], int node, int cluster[], int *size) {
    visited[node] = true;
    cluster[(*size)++] = node;
    
    for (int i = 0; i < WORM_COUNT; i++) {
        if (adjMatrix[node*WORM_COUNT + i] && !visited[i]) {
            dfs(adjMatrix, visited, i, cluster, size);
        }
    }
}

// Function to find and print clusters
void find_clusters(bool *adjMatrix, int* clusters) {
    bool visited[WORM_COUNT] = {false};
    int cluster_counter = 0;
    
    for (int i = 0; i < WORM_COUNT; i++) {
        if (!visited[i]) {
            int cluster[WORM_COUNT]; // Temporary storage for cluster elements
            int size = 0;
            
            dfs(adjMatrix, visited, i, cluster, &size);

            if (size > 1){
                clusters[cluster_counter] = size;
                cluster_counter += 1;
            }
        }
    }
}

int get_adjacency_matrix(bool* adjacency_matrix, int worm_count, float* positions, int t, float r){
    float diff_x, diff_y, dist = 0;
    for (int j = 0; j < worm_count; ++j){
        for (int k = 0; k < worm_count; ++k){
            diff_x = (positions[(t * worm_count + j) * 2] - positions[(t * worm_count + k) * 2]);
            diff_y = (positions[(t * worm_count + j) * 2 + 1] - positions[(t * worm_count + k) * 2 + 1]);
            dist = sqrt(diff_x * diff_x + diff_y * diff_y);
            if (dist <= r){
                adjacency_matrix[j*WORM_COUNT + k] = true;
            }
            else{
                adjacency_matrix[j*WORM_COUNT + k] = false;
            }
        }
    }
    return 0;
}

int reset_matrix(bool* adjacency_matrix){
    for (int i = 0; i < WORM_COUNT; ++i){
        for (int j = 0; j < WORM_COUNT; ++j){
            adjacency_matrix[i*WORM_COUNT + j] = false;
        }
    }
    return 0;
}


int main(int argc, char* argv[]) {

    std::string pos_file = std::string(argv[1]).append(argv[2]).append("_pos.dat");
    // std::string pos_file = std::string(argv[1]).append(argv[2]).append("/").append(argv[3]).append("_pos.dat");

    int timesteps = (int) (N_STEPS / LOGGING_INTERVAL);    
    float *data = loadCompressed(pos_file, WORM_COUNT * timesteps * 2);

    float *results = new float[(timesteps / 10) * WORM_COUNT + (timesteps / 10) + 1];
    // float *results = new float[1];
    // float *results = new float[(timesteps / 10) * WORM_COUNT];


    // track clusters
    float cluster_size = 0, elong = 0;
    int tot_n = 0, nn = 0;
    bool* adjacency_matrix = new bool[WORM_COUNT * WORM_COUNT];

    for (int t = 0; t < timesteps; t += 10){
        reset_matrix(adjacency_matrix);
        get_adjacency_matrix(adjacency_matrix, WORM_COUNT, data, t, CLUSTERING_RADIUS);
        int clusters[WORM_COUNT] = {0};
        find_clusters(adjacency_matrix, clusters);
        for (int c = 0; c < WORM_COUNT; c++){
            results[(t / 10) * WORM_COUNT + c] = clusters[c];
        }

        reset_matrix(adjacency_matrix);
        get_adjacency_matrix(adjacency_matrix, WORM_COUNT, data, t, NEIGHBOR_RADIUS);
        elong = 0;
        tot_n = 1;
        for (int j = 0; j < WORM_COUNT; j++){
            // results[j] = 0;
            float agent_x = data[(t * WORM_COUNT + j) * 2];
            float agent_y = data[(t * WORM_COUNT + j) * 2 + 1];
            if (!(agent_x == WIDTH || agent_x == 0 || agent_y == HEIGHT || agent_y == 0)){
                nn = 0;
                for (int k = 0; k < WORM_COUNT; k++){
                    if (adjacency_matrix[j * WORM_COUNT + k]){
                        nn += 1;
                    }
                }       
                if (nn > 5){ 
                    //float *npos = new float[nn * 2];
                    std::vector<float> npos(nn * 2);
                    int idn = 0;
                    for (int k = 0; k < WORM_COUNT; k++){
                        if (adjacency_matrix[j * WORM_COUNT + k]){
                            npos[idn * 2] = data[((WORM_COUNT * t) + k) * 2];
                            npos[idn * 2 + 1] = data[((WORM_COUNT * t) + k) * 2 + 1];
                            idn += 1;
                        }
                    }

                    // Compute distances and create sorted indices
                    std::vector<std::pair<double, size_t>> distance_index_pairs;
                    distance_index_pairs.reserve(npos.size());
                    
                    for (size_t i = 0; i < nn; ++i) {
                        double dx = npos[i * 2] - agent_x;
                        double dy = npos[i * 2 + 1] - agent_y;
                        double dist = std::sqrt(dx * dx + dy * dy);
                        distance_index_pairs.push_back({dist, i});
                    }

                    // Sort by distance
                    std::sort(distance_index_pairs.begin(), distance_index_pairs.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });
                    
                    // Compute gaps between consecutive distances
                    std::vector<double> gaps;
                    gaps.reserve(distance_index_pairs.size() - 1);
                    
                    for (size_t i = 1; i < distance_index_pairs.size(); ++i) {
                        gaps.push_back(distance_index_pairs[i].first - distance_index_pairs[i-1].first);
                    }
                    
                    // Compute mean gap
                    double mean_gap = 0.0;
                    for (double gap : gaps) {
                        mean_gap += gap;
                    }
                    mean_gap /= gaps.size();

                    // Find first large gap
                    size_t cutoff_idx = distance_index_pairs.size();
                    for (size_t i = 0; i < gaps.size(); ++i) {
                        if (gaps[i] > 10.0 * mean_gap) {
                            cutoff_idx = i + 1;
                            break;
                        }
                    }

                    // Check if we have enough valid neighbors
                    if (cutoff_idx >= 10) {
                        std::vector<float> valid_neighbors(cutoff_idx * 2);

                        for (size_t i = 0; i < cutoff_idx; ++i) {
                            size_t idx = distance_index_pairs[i].second;
                            valid_neighbors[i * 2] = npos[idx * 2];
                            valid_neighbors[i * 2 + 1] = npos[idx * 2 + 1];
                        }            

                        std::vector<Vector2d> points = convert1DArrayTo2DVector(valid_neighbors, cutoff_idx);
                        Matrix2d cov = computeRegularizedCovariance(points, 1e-1);
                        SelfAdjointEigenSolver<Matrix2d> solver(cov);
                        Vector2d eigenvalues = solver.eigenvalues();
                        eigenvalues[0] = std::max(eigenvalues[0], 1e-2);
                        eigenvalues[1] = std::max(eigenvalues[1], 1e-2);
                        if (eigenvalues[0] > eigenvalues[1] & eigenvalues[1] > 0) {
                            // results[j] = (eigenvalues[0] / eigenvalues[1]);
                            // std::cout << eigenvalues[0] << "/" << eigenvalues[1] << std::endl;
                            elong += cutoff_idx * (eigenvalues[0] / eigenvalues[1]);
                        }
                        else if (eigenvalues[0] > 0) {
                            // results[j] = (eigenvalues[1] / eigenvalues[0]);
                            // std::cout << eigenvalues[1] << "/" << eigenvalues[0] << std::endl;
                            elong += cutoff_idx * (eigenvalues[1] / eigenvalues[0]);  
                        }

                        tot_n += cutoff_idx;
                    }
                    else tot_n += cutoff_idx;
                    //delete [] npos;
                }
            }
            else tot_n += nn;
        } 
        elong /= (tot_n); 
        results[(timesteps / 10) * WORM_COUNT + t / 10] = elong;
    }

    free(adjacency_matrix);

    // compute mean squared displacement
    float time_averaged_msd = 0, mean_squared_disp, diff_x, diff_y, sq_dist = 0;
    for (int t = 0; t < ((int)(N_STEPS / LOGGING_INTERVAL)) - MSD_WINDOW; t += MSD_WINDOW){
        mean_squared_disp = 0;
        for (int i = 0; i < WORM_COUNT; ++i){
            diff_x = (data[((t + MSD_WINDOW) * WORM_COUNT + i) * 2] - data[(t * WORM_COUNT + i) * 2]);
            if (diff_x > SPEED * MSD_WINDOW / DT){
                diff_x -= WIDTH;
            }
            if (diff_x < -SPEED * MSD_WINDOW / DT){
                diff_x += WIDTH;
            }            
            diff_y = (data[((t + MSD_WINDOW) * WORM_COUNT + i) * 2 + 1] - data[(t * WORM_COUNT + i) * 2 + 1]);
            if (diff_y > SPEED * MSD_WINDOW / DT){
                diff_y -= HEIGHT;
            }
            if (diff_y < -SPEED * MSD_WINDOW / DT){
                diff_y += HEIGHT;
            }            
            sq_dist = diff_x * diff_x + diff_y * diff_y;
            mean_squared_disp += sq_dist;
            if (mean_squared_disp != mean_squared_disp){
                std::cout << t << " " << i << std::endl;
                return 1;
            }
        }
        time_averaged_msd += (mean_squared_disp / WORM_COUNT);
    }
    time_averaged_msd /= (int)(N_STEPS / LOGGING_INTERVAL / MSD_WINDOW); 
    results[(timesteps / 10) * WORM_COUNT + (timesteps / 10)] = time_averaged_msd; 
    // results[0] = time_averaged_msd; 

    std::string res_file = std::string(argv[1]).append(argv[2]).append("_res.dat");
    // std::string res_file = std::string("/media/carlo/HD2/res_half/").append("res/").append(argv[2]).append("_").append(argv[3]).append("_msd.dat");
    saveCompressed(results, (timesteps / 10) * WORM_COUNT + (timesteps / 10) + 1, res_file);
    // saveCompressed(results, 1, res_file);
    
    return 0;
}