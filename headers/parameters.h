// config.h

// Environmental variables
#define N 128
#define WIDTH 60.0f
#define HEIGHT 60.0f

// Simulation parameters
#define WORM_COUNT 20
#define N_STEPS 1000
#define LOGGING_INTERVAL 1
#define DT 1.0f
#define DEBUG false
#define ENABLE_RANDOM_INITIAL_POSITIONS true
#define INITIAL_AREA_NUMBER_OF_CELLS 10
#define ENABLE_MAXIMUM_NUMBER_OF_AGENTS_PER_CELL false
#define MAXIMUM_AGENTS_PER_CELL 40
#define LOG_VELOCITIES true
#define LOG_ANGLES true
#define LOG_POTENTIAL true
#define LOG_TRAJECTORIES true
#define LOG_GRID true
#define LOG_PHEROMONES false
#define LOG_AGENT_COUNT_GRID false

// Agent parameters
#define SPEED 0.1f
#define SENSING_RANGE 1
#define ODOR_THRESHOLD 1e-8f
#define ON_FOOD_SPEED_SCALE 0.08f// scale = exp(mu) => mu = log(scale)
#define ON_FOOD_SPEED_SHAPE 0.45f// shape = sigma
#define OFF_FOOD_SPEED_SCALE 0.09f// scale = exp(mu) => mu = log(scale)
#define OFF_FOOD_SPEED_SHAPE 0.63f // shape = sigma
#define PIROUETTE_TO_RUN_THRESHOLD 1e-6f
#define AUTO_TRANSITION_PROBABILITY_THRESHOLD 0.01f
#define KAPPA 11.0f
#define MAX_ALLOWED_SPEED 0.3f

// Odor parameters
#define TARGET_AREA_SIDE_LENGTH 20
#define MAX_CONCENTRATION 10.0f
#define GAMMA 0.0001f
#define DIFFUSION_CONSTANT 0.05f
#define ATTRACTION_STRENGTH 0.0282f
#define ATTRACTION_SCALE 1.0f

// Pheromone parameters
#define ATTRACTANT_PHEROMONE_SCALE 15.0f
#define ATTRACTANT_PHEROMONE_STRENGTH 0.0f//0.000282f
#define ATTRACTANT_PHEROMONE_DECAY_RATE 0.001f
#define ATTRACTANT_PHEROMONE_SECRETION_RATE 0.00001f
#define ATTRACTANT_PHEROMONE_DIFFUSION_RATE 0.0001f
#define REPULSIVE_PHEROMONE_SCALE 15.0f
#define REPULSIVE_PHEROMONE_STRENGTH 0.0f//(-0.0000031f)
#define REPULSIVE_PHEROMONE_DECAY_RATE 0.0001f
#define REPULSIVE_PHEROMONE_SECRETION_RATE 0.00001f
#define REPULSIVE_PHEROMONE_DIFFUSION_RATE 0.00001f

// Noise parameters
#define SIGMA 0.0f
#define ENVIRONMENTAL_NOISE 0.0f

// CUDA parameters
#define BLOCK_SIZE 32

__constant__ float DX = WIDTH/N;
__constant__ float DY = HEIGHT/N;