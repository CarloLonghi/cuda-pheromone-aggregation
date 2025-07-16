// config.h

// Environmental variables
#define N 1024
#define WIDTH 50.0f
#define HEIGHT 50.0f
#define SEED 1234
// Simulation parameters
#define WORM_COUNT 200
#define MAX_WORMS 200
#define TIME 600
#define DT 0.1f
#define N_STEPS int(TIME / DT)
#define LOGGING_INTERVAL int(1 / DT)
#define DEBUG false
#define ENABLE_RANDOM_INITIAL_POSITIONS false
#define INITIAL_AREA_SIZE 5.0f
#define ENABLE_MAXIMUM_NUMBER_OF_AGENTS_PER_CELL true
#define MAXIMUM_AGENTS_PER_CELL 40
#define LOG_POTENTIAL false
#define LOG_GRID false
#define LOG_PHEROMONES false
#define LOG_AGENT_COUNT_GRID false
#define LOG_GENERIC_TARGET_DATA false
#define LOG_POSITIONS true
#define LOG_ANGLES false
#define LOG_VELOCITIES false

// Agent parameters

#define N_STATES 7

#define BODY_LENGTH 0.5f

#define SENSING_RADIUS 0.1f
#define SPEED 0.15f
#define PHEROMONE_THRESHOLD 0.0
#define MAX_CONCENTRATION 1.0 // of the pheromone

#define ALIGNMENT_RADIUS BODY_LENGTH
#define ALIGNMENT_STRENGTH 0.05f // 0.05f
#define SLOWING_FACTOR 5.0f // 20.0f

// Descriptor parameters
#define CLUSTERING_RADIUS 2 * BODY_LENGTH
#define NEIGHBOR_RADIUS 0.5f
#define MSD_WINDOW 50

// Noise parameters
#define SIGMA 1e-8f
#define ENVIRONMENTAL_NOISE 0.0f
#define SENSING_NOISE 0.00f // 0.01

// Odour parameters
#define MU_X 5.0f      // Mean x of the Gaussian
#define MU_Y 30.0f      // Mean y of the Gaussian
#define A 0.5f         // Amplitude of the Gaussian
#define SIGMA_X 5.0f   // Standard deviation in x direction
#define SIGMA_Y 5.0f   // Standard deviation in y direction
#define TARGET_AREA_SIDE_LENGTH 40
#define ODOUR_MAX_CONCENTRATION 1.0f
#define GAMMA 0.0001f
#define DIFFUSION_CONSTANT 0.005f //more than 0.01, it explodes

// CUDA parameters
#define BLOCK_SIZE 32

__constant__ float DX = WIDTH/N;
__constant__ float DY = HEIGHT/N;