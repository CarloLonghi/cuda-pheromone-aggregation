
# Massive Multi-Agent Worm Simulator (MMA-WORMSIM)

The following project is an implementation of an agent-based simulator of C. *elegans*, part of the [BABOTs project](www.babots.eu). In light of the project's aim at instilling collective behaviors within the worms, the simulator is tailored to allow a large number of agents in parallel interacting with a number of stimuli. For this reason, we seek the simplest possible representation of a worm, e.g. a 2D point moving in space, while still validating their movement to match reality. In particular, we aim at developing the following behaviors/interactions:

| Behavior | Status |
| ----------- | ----------- |
| Search | To be validated |
| Chemotaxis | To be validated |
| Pheromones | To be validated |
| Density | To be developed |


Currently, the search behavior of C. *elegans* is implemented as per [Salvador et al., 2014](https://royalsocietypublishing.org/doi/10.1098/rsif.2013.1092), thus each agent is equipped with a set of sub-states: 3 crawling states (I merged open and closed arcs) and 4 reorientation sub-states. The word "sub-state" will refer to these sub-states in the following. The transition probabilities to switch between sub-states are evinced from the paper. I am in the process of validating the parameters to match those of real worms. Chemotaxis has been implemented according to [Tanimoto et al., 2017](https://elifesciences.org/articles/21629), in particular taking into account that an agent can either be in a high turning state (pirouette) or in a low turning state (run). In the following, the word "state" refers to these two states. This still needs to be validated, in particular how the parameters of the attractant interact with a worm. The diffusion of the chemical odor is implemented both with a Gaussian diffusion model (for simplicity) and a the numerical solution of the partial differential equation (PDE) of diffusion via a finite different scheme, which is currently disabled. Attractive and repulsive pheromones are implemented as well through the finite difference method on PDEs based on [Avery et al., 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009231), which is disabled too. Enabling these behaviors is explained in the next section. Density-induces behavior will be implemented in the future.

All the stimuli are grouped into a potential, representing the perceptual stimuli acting on a single agent at a given location, similarly to [Avery et al., 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009231), where they only use the potentials relative to the attractive and repulsive pheromones, together with a "squeezing" potential. The sum of potentials is referred to as "potential" from now on for the sake of brevity.

#Structure
This project is a work in progress, thus discrepancies between the following description and the actual code may arise.

##Environment
A square environment with periodic boundary conditions.

##Data structures
Currently, only 2 data structures (structs) are used:
-`Agent`: represents a 2D point in space with `x` and `y` coordinates, the current absolute bearing `angle` (between -pi and +pi), the current velocity `speed`, the previously sensed potential `previous_potential`, the cumulative sum of potentials in the previous steps where the agent was in the "pirouette" state ([Tanimoto et al., 2017](https://elifesciences.org/articles/21629)) `cumulative_potential`, its current state `state` (0=run, 1=pirouette), its current sub-state id `sub_state`, its previous sub-state `previous_substate` and some data relative to chemotactic experiments, namely the number of timesteps in the target area and the first time it entered the target area.
-`ExplorationState`: represents the sub-state of an agent with an `id` (0=loop, 1=arc, 2=line, 3=pirouette, 4=omega, 5=reversal, 6=pause), parameters relative to the speed log normal distribution of an agent in this sub-state (`speed_scale`, `speed_spread`), the von mises angle distribution (`angle_mu`, `angle_kappa`), the parameters of the lognormal distribution to find the number of time steps to spend within a sub-state (`duration_mu`, `duration_sigma`), the number of timesteps the agent has spent in the sub-state, the actual duration that has been drawn from the lognormal, the maximum allowed duration and the sign of the angle picked (since the distribution is lognormal, it is always positive, thus this sign gives a 50-50 chance of picking a negative angle, effectively allowing the agents to turn both left and right), a list of transition `probabilities` from this sub-state to the others, including itself. Self-loops are allowed only on crawl states.

The rest are static arrays, such as the array of Agents and the grids of odor/pheromone concentration.

##Code logic

The `main.cu` contains the initialization, main loop and logging of the simulation. The initialization is done by either reading the parameters through `argv` or by reading a `.json` file. If you wish to use default parameters, the initialization should be done by calling `initProbabilities(h_explorationStates);` (instead of `initProbabilitiesWithParams`). The main loop is composed of moving the agents, updating the probabilities of the finite state machine they use to explore and logging their positions, speed, angle and state. Next, the grids relative to the chemicals are updated, if present, and logged as well. Finally, the logged data is saved to a `agents_all_data.json` file. In the following, I describe the headers.

`agent_update.h`:
-`moveAgents` is the CUDA kernel responsible for determining the next position of an agent. It first looks for a chemical gradient around the agent by randomly sampling its surrounding within a `SENSING_RADIUS` and finds the direction corresponding to the maximum sensed chemical with some optional perceptual noise (Gaussian). Next, it picks the current sub-state's distribution probabilities and samples an angle and a speed. If the agent is in the exploration state (thus not climbing a gradient), the angle and speed correspond to those sampled. Otherwise, it will choose the direction of maximum potential and a speed sampled from a lognormal distribution, which is for now fitted to averaged experimental data and still needs validation. Next, the sub-state update logic is implemented to stochastically pick the next state, unless the current sub-state is in a self-loop: in this case it will remain for the sub-state's `duration` and decrease it.

`gaussian_odour.h`: calculates the concentration of odor based on a Gaussian diffusion model.

`logging.h`: saves all parameters and data generated by the execution of the code to a `.json`. The main function to look at is `saveAllDataToJSON`.

`init_env.h`: contains the structs and the initialisation of the agents, of the grids and of the sub-states.

`numeric_functions.h`: a collection of functions for the numerical solution of PDEs.

`parameters.h`: all the parameters of the simulation. 

`update_matrices.h`: functions that update the grids of chemical odor and pheromones, together with the grid of potentials.


#Parameters

- `N`	: quantization of the environment for the finite difference scheme; higher than 256 doesn't work well, needs optimisation of the block size of CUDA
- `WIDTH`	: width of the environment in mm
- `HEIGHT`: height of the environment in mm
- `SEED` 	: ideally, seed for all the random number generators; realistically, only used in a couple of functions, should be put everywhere.
- `WORM_COUNT`: number of agents
- `N_STEPS`: number of time steps of the simulation
- `LOGGING_INTERVAL`: the number of time steps separating the logging of grids, useful to save space and time (logging grids is the MOST time consuming part of the code)
- `DT`: time interval, used for the Euler scheme together with finite difference
- `DEBUG`: adds some prints
- `ENABLE_RANDOM_INITIAL_POSITIONS`: tells the program if the agents positions should be at random or within some initial area
- `INITIAL_AREA_NUMBER_OF_CELLS`: if above is true, this is the length of the side of the initial area
- `ENABLE_MAXIMUM_NUMBER_OF_AGENTS_PER_CELL`: tells the program if there should be a limit to the density of agents inside a cell (a grid is composed of cells)
- `MAXIMUM_AGENTS_PER_CELL`: if above is true, this is the maxmimum number of agents per cell
- `LOG_POTENTIAL`: unused, tells the program whether to log the grid of potentials every `LOGGING_INTERVAL` time step(s)
- `LOG_PHEROMONES`: unused, tells the program whether to log the grids of attractive and repulsive pheromones every `LOGGING_INTERVAL` time step(s)
- `LOG_AGENT_COUNT_GRID`: unused, tells the program whether to log the grid with the number of agents within each cell every `LOGGING_INTERVAL` time step(s)
- `LOG_GENERIC_TARGET_DATA`: tells the program whether to log the agents' data (position, angle, speed, sub-states and other parameters)
- `LOG_POSITIONS, LOG_ANGLES, LOG_VELOCITIES`: tells the program whether to log, respectively, position, angle and speed of every agent at every timestep
- `N_STATES`: number of sub-states
- `SENSING_RADIUS`: maximum radius to sense a potential in mm
- `SPEED`: initial speed of an agent
- `SENSING_RANGE`: unused, same as `SENSING_RADIUS` but works in terms of cells rather than absolute mm
- `ODOR_THRESHOLD`: minimum potential sensed by an agent
- `ON_FOOD_SPEED_SCALE, ON_FOOD_SPEED_SHAPE`: unused, average lognormal speed distribution parameters
- `OFF_FOOD_SPEED_SCALE_SLOW, OFF_FOOD_SPEED_SHAPE_SLOW, OFF_FOOD_SPEED_SLOW_WEIGHT, OFF_FOOD_SPEED_SCALE_FAST, OFF_FOOD_SPEED_SHAPE_FAST`: parameters of the lognormal mixtures for the speed when the agent doesn't sense potential. Fitted to average data, unused. 
- `ON_FOOD_SPEED_SCALE_SLOW, ON_FOOD_SPEED_SHAPE_SLOW, ON_FOOD_SPEED_SLOW_WEIGHT, ON_FOOD_SPEED_SCALE_FAST, ON_FOOD_SPEED_SHAPE_FAST`: parameters of the lognormal mixtures for the speed when the agent senses potential. Fitted to average data, unused. 
- `LOOP_TIME_MU, LOOP_TIME_SIGMA, ARC_TIME_MU, ARC_TIME_SIGMA, LINE_TIME_MU, LINE_TIME_SIGMA`: parameters of the lognormal distributions for the duration (in seconds) of sub-states 0-2 (loop, arc, line). Fitted to average data.
- `PIROUETTE_TO_RUN_THRESHOLD`: threshold to go from state 1 to 0 ([Tanimoto et al., 2017](https://elifesciences.org/articles/21629))
- `AUTO_TRANSITION_PROBABILITY_THRESHOLD`: the probability of perfoming a spontaneous transition from a state to the other (NOT SUB-STATES)
- `KAPPA`: unused
- `MAX_ALLOWED_SPEED`: maximum speed for an agent
- `MU_X, MU_Y, A, SIGMA_X, SIGMA_Y`: parameters for the Gaussian odor
- `TARGET_AREA_SIDE_LENGTH`: length of the side of the square where the odor is placed
- `MAX_CONCENTRATION`: initial and maximum concentration of odor.
- `GAMMA`: odor evaporation rate
- `DIFFUSION_CONSTANT`: odor diffusion constant
- `ATTRACTION_STRENGTH, ATTRACTION_SCALE`: parameters to transfrom the concentration into a potential as per [Avery et al., 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009231)
- `ATTRACTANT_PHEROMONE_SCALE, ATTRACTANT_PHEROMONE_STRENGTH, ATTRACTANT_PHEROMONE_DECAY_RATE, ATTRACTANT_PHEROMONE_SECRETION_RATE, ATTRACTANT_PHEROMONE_DIFFUSION_RATE`: same parameters as the odor, but for the attractive pheromone, which has a secretion rate;
- `REPULSIVE_PHEROMONE_SCALE, REPULSIVE_PHEROMONE_STRENGTH, REPULSIVE_PHEROMONE_DECAY_RATE, REPULSIVE_PHEROMONE_SECRETION_RATE, REPULSIVE_PHEROMONE_DIFFUSION_RATE`: parameters for the repulsive pheromones, same as the attractive pheromones
- `SIGMA`: standard deviation of the white perceptual noise
- `ENVIRONMENTAL_NOISE`: standard deviation of the white environmental noise
- `BLOCK_SIZE`: size of a CUDA block

#Visualization

The simulator does not support real-time simulation, it only writes the history of "what happened" to a json. We provide a simple example Python implementation of a video renderer for the simulator, which allows only Gaussian odor, and a few functions to animate 1 grid (useful for visualizing the potential) and multiple grids (useful to debug different grids). The reason why there is not real-time visualization is two-fold: on one hand, the complexity of it, on the other the little improvement it would give in terms of time saving. 


# Installation

The following guide is written for Nobara 40, similar steps are to be executed on other Linux distros. I do not grant support for Windows/MacOS.

## Prerequisites

Ensure you have the following software installed before proceeding:
- **CUDA Toolkit**: [12.3](https://developer.nvidia.com/cuda-12-3-0-download-archive)
- **GCC**: 11.4.0 (I recommend installing [Homebrew for Linux](https://docs.brew.sh/Homebrew-on-Linux)
## Setting Up Environment Variables

### Step 1: Verify Installed Versions

You can verify the installed versions of `gcc` and `nvcc` using the following commands:

```bash
gcc --version
nvcc --version
```

Make sure the versions match the ones required for the project.

### Step 2: Setting the Environment Variables

Add the following lines to your shell configuration file (e.g., `.bashrc`, `.zshrc`) to set the required environment variables:

```bash
# Set environment variables for GCC and NVCC
export PATH=/usr/local/cuda-X.X/bin:$PATH  # Update with the correct CUDA version
export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-X.X                 # Update with your GCC path and version
export CXX=/usr/bin/g++-X.X                # Update with your G++ path and version
```

After modifying the file, apply the changes by running:

```bash
source ~/.bashrc  # or ~/.zshrc depending on your shell
```

### Step 3: Verify the Setup

To confirm that the environment variables are set correctly, run:

```bash
echo $PATH
echo $LD_LIBRARY_PATH
echo $CC
echo $CXX
```

### Step 4: Compiling the CUDA Simulator

Once the environment is configured, compile the CUDA simulator by navigating to the project directory and running the following command:

```bash
nvcc main.cu -o main -lm -lstdc++ --expt-relaxed-constexpr
```

#CLion Setup
In the following `cuda_dir` refers to the NVCC from CUDA Toolkit v12.3 insallation directory, which on most Linux distributions is installed in , while `gcc_dir` refers to the directory where GCC v11.4 is installed. On most Linux systems, the former is something like `/usr/local/cuda-12.3/bin/nvcc`, while the latter, if installed through Homebrew looks like `/home/linuxbrew/.linuxbrew/opt/gcc@11/bin/`.
If you are using CLion, you can build and run the project by accessing `Settings > Build, Execution, Deployment > Toolchains` and setting the "C Compiler" and "C++ Compiler" to the binaries of the respective compilers, which are in the `gcc_dir` and named `gcc-11` and `g++11` respectivly. Then, go to `Settings > Build, Execution, Deployment > CMake` and add the following CMake options: `-DCMAKE_CUDA_COMPILER=cuda_dir  -DCMAKE_CUDA_FLAGS="--compiler-bindir=gcc_dir/gcc-11"`. Now you should be able to build and execute the project.



