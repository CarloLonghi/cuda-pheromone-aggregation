// config.h

// Environmental variables
#define NN 256
#define WIDTH 128.0f
#define HEIGHT 128.0f
// Simulation parameters
#define WORM_COUNT 10000
#define TIME 200
#define DT 0.1f
#define N_STEPS int(TIME / DT)
#define LOGGING_INTERVAL 10
#define DEBUG false
#define ENABLE_RANDOM_INITIAL_POSITIONS false
#define INITIAL_AREA_SIZE 10.0f
#define LOG_POTENTIAL false
#define LOG_GRID false
#define LOG_PHEROMONES false
#define LOG_AGENT_COUNT_GRID false
#define LOG_GENERIC_TARGET_DATA false
#define LOG_POSITIONS true
#define LOG_ANGLES true
#define LOG_VELOCITIES false

// Agent parameters
#define BODY_LENGTH 0.25f
#define SPEED 0.4f * DT //(0.3f * BODY_LENGTH * DT)
#define MAX_CONCENTRATION 1.0 // of the pheromone
#define ALIGNMENT_RADIUS 1.0f
#define REPULSION_RADIUS (ALIGNMENT_RADIUS*0.3f)

// Descriptor parameters
#define CLUSTERING_RADIUS (2 * BODY_LENGTH)
#define NEIGHBOR_RADIUS 0.5f
#define MSD_WINDOW 10
#define ARM_RANGE (BODY_LENGTH * 2.0)

// Noise parameters
#define SIGMA 0.1f // 0.015f
#define GAMMA (SPEED * 0.1f)
#define ENVIRONMENTAL_NOISE  1.0f

// Odour parameters
#define MU_X 5.0f      // Mean x of the Gaussian
#define MU_Y 25.0f      // Mean y of the Gaussian
#define DIFFUSION_CONSTANT 0.0005f //more than 0.01, it explodes

// CUDA parameters
#define BLOCK_SIZE 32

__constant__ float DX = WIDTH/NN;
__constant__ float DY = HEIGHT/NN;
