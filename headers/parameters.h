// config.h

// Environmental variables
#define NN 1024
#define WIDTH 50.0f
#define HEIGHT 50.0f
#define SEED 1234
// Simulation parameters
#define WORM_COUNT 200
#define TIME 600
#define DT 0.1f
#define N_STEPS int(TIME / DT)
#define LOGGING_INTERVAL int(1 / DT)
#define DEBUG false
#define ENABLE_RANDOM_INITIAL_POSITIONS false
#define INITIAL_AREA_SIZE 2.5f
#define LOG_POTENTIAL false
#define LOG_GRID false
#define LOG_PHEROMONES false
#define LOG_AGENT_COUNT_GRID false
#define LOG_GENERIC_TARGET_DATA false
#define LOG_POSITIONS true
#define LOG_ANGLES false
#define LOG_VELOCITIES false

// Agent parameters
#define BODY_LENGTH 0.5f
#define SPEED 0.15f * DT
#define MAX_CONCENTRATION 1.0 // of the pheromone
#define ALIGNMENT_RADIUS BODY_LENGTH
#define REPULSION_RADIUS (BODY_LENGTH * 0.2)

// Descriptor parameters
#define CLUSTERING_RADIUS (2 * BODY_LENGTH)
#define NEIGHBOR_RADIUS 0.5f
#define MSD_WINDOW 50

// Noise parameters
#define SIGMA 1e-8f
#define ENVIRONMENTAL_NOISE 0.0f
#define SENSING_NOISE 0.00f // 0.01

// Odour parameters
#define MU_X 5.0f      // Mean x of the Gaussian
#define MU_Y 25.0f      // Mean y of the Gaussian
#define DIFFUSION_CONSTANT 0.0005f //more than 0.01, it explodes

// CUDA parameters
#define BLOCK_SIZE 32

#define CELL_SIZE  (2.0f * ALIGNMENT_RADIUS)  // Size should be ≥ interaction range
#define GRID_DIM_X (int)ceil(WIDTH / CELL_SIZE)
#define GRID_DIM_Y (int)ceil(HEIGHT / CELL_SIZE)

__constant__ float DX = WIDTH/NN;
__constant__ float DY = HEIGHT/NN;
