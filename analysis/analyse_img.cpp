#include <fstream>
#include <vector>
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include <stdio.h>
#include <zlib.h>

using namespace Eigen;

#define TIME 300
#define DT 0.1f
#define N_STEPS int(TIME / DT)
#define LOGGING_INTERVAL 10
#define WORM_COUNT 10000
#define WIDTH 50
#define HEIGHT 50

// Agent parameters
#define BODY_LENGTH 0.25f
#define SPEED (0.3f * BODY_LENGTH * DT)
#define MAX_CONCENTRATION 1.0 // of the pheromone
#define ALIGNMENT_RADIUS (2 * BODY_LENGTH)
#define REPULSION_RADIUS BODY_LENGTH

// Descriptor parameters
#define CLUSTERING_RADIUS (0.5 * BODY_LENGTH)
#define NEIGHBOR_RADIUS (4.0 * BODY_LENGTH)
#define MSD_WINDOW 2
#define ARM_RANGE (BODY_LENGTH * 2.0)

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

float* read_compressed_floats_with_header(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }
    
    // Read array size from header
    uint64_t array_size;
    file.read(reinterpret_cast<char*>(&array_size), sizeof(uint64_t));
    
    // Read the compressed data
    file.seekg(0, std::ios::end);
    size_t compressed_size = file.tellg() - sizeof(uint64_t);
    file.seekg(sizeof(uint64_t), std::ios::beg);
    
    std::vector<Bytef> compressed(compressed_size);
    file.read(reinterpret_cast<char*>(compressed.data()), compressed_size);
    
    // Prepare decompression buffer
    float* data = new float[array_size];
    uLongf decompressed_size = array_size * sizeof(float);
    
    // Decompress
    int result = uncompress(
        reinterpret_cast<Bytef*>(data), &decompressed_size,
        compressed.data(), compressed.size()
    );
    
    if (result != Z_OK) {
        std::cerr << "Decompression failed with error: " << result << std::endl;
        return {};
    }
    
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
int find_clusters(bool *adjMatrix) {
    bool visited[WORM_COUNT] = {false};
    int biggest_size = 0;
    
    for (int i = 0; i < WORM_COUNT; i++) {
        if (!visited[i]) {
            int cluster[WORM_COUNT]; // Temporary storage for cluster elements
            int size = 0;
            
            dfs(adjMatrix, visited, i, cluster, &size);

            if (size > biggest_size){
                biggest_size = size;
            }
        }
    }
    return biggest_size;
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

    std::string pos_file = "./json/elong_test.dat";

    float *data = read_compressed_floats_with_header(pos_file);

    float r = std::stof(argv[2]);

    // track clusters
    float cluster_size = 0, elong = 0;
    int tot_n = 0, nn = 0;
    // bool* adjacency_matrix = (bool*)malloc(WORM_COUNT * WORM_COUNT * sizeof(bool));
    bool* adjacency_matrix = new bool[WORM_COUNT * WORM_COUNT];
    reset_matrix(adjacency_matrix);
    get_adjacency_matrix(adjacency_matrix, WORM_COUNT, data, 0, 10);
    float cluster = find_clusters(adjacency_matrix);

    float *res = new float[2];
    res[0] = cluster;

    reset_matrix(adjacency_matrix);
    get_adjacency_matrix(adjacency_matrix, WORM_COUNT, data, 0, r);
    elong = 0;
    tot_n = 0;
    for (int j = 0; j < WORM_COUNT; j++){
        // results[j] = 0;
        float agent_x = data[j * 2];
        float agent_y = data[j * 2 + 1];
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
                    npos[idn * 2] = data[k * 2];
                    npos[idn * 2 + 1] = data[k * 2 + 1];
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
            if (cutoff_idx >= 5) {
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
        else tot_n += nn;
    } 
    elong /= (tot_n); 
    res[1] = elong;

    free(adjacency_matrix);

    std::string res_file = std::string("./img_data/").append(argv[1]).append("_").append(argv[2]).append("_").append(argv[3]);

    saveCompressed(res, 2, res_file);

    // std::cout << cluster << " " << elong << std::endl;
    
    return 0;
}