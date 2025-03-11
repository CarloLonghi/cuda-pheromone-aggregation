import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

def load_and_animate_agents_and_grid2(json_file_path, fps, dest_file_path="animation.mp4"):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    parameters = data['parameters']
    N = parameters['N']
    LOGGING_INTERVAL = parameters['LOGGING_INTERVAL']
    N_STEPS = parameters['N_STEPS']
    WIDTH = parameters['WIDTH']
    HEIGHT = parameters['HEIGHT']
    print(parameters)

    sub_states = [[data["sub_states"][agent][timestep] for timestep in range(int(N_STEPS // LOGGING_INTERVAL))] for agent in range(N)]
    sub_states_map = {
        0: "Loop",
        1: "Arc",
        2: "Line",
        3: "Pirouette",
        4: "Omega",
        5: "Reversal",
        6: "Pause"
    }

    # Prepare the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)

    # Create a list of scatter plot objects for each agent
    scatters = [ax.plot([], [], 'o', color='white', markersize=1.0)[0] for _ in range(N)]
    traces = [ax.plot([], [], '-', color='white', linewidth=0.5)[0] for _ in range(N)]
    position_matrix = [[data["positions"][agent][timestep] for timestep in range(int(N_STEPS // LOGGING_INTERVAL))] for agent in range(N)]

    # Parameters for the evolving Gaussian density function
    MU_X = parameters['MU_X']
    MU_Y = parameters['MU_Y']
    A = parameters['A']
    GAMMA = parameters['GAMMA']
    SIGMA_X = parameters['SIGMA_X']
    SIGMA_Y = parameters['SIGMA_Y']
    DIFFUSION_CONSTANT = parameters['DIFFUSION_CONSTANT']
    ATTRACTION_STRENGTH = parameters['ATTRACTION_STRENGTH']
    ATTRACTION_SCALE = parameters['ATTRACTION_SCALE']
    MAX_CONCENTRATION = parameters['MAX_CONCENTRATION']

    # Function to calculate the evolving Gaussian density
    def calculate_gaussian_density(t, X, Y):
        dx = X - MU_X
        dy = Y - MU_Y
        a_t = A * np.exp(-GAMMA * t)
        sigma_x_t = SIGMA_X + 2 * DIFFUSION_CONSTANT * t
        sigma_y_t = SIGMA_Y + 2 * DIFFUSION_CONSTANT * t
        density = MAX_CONCENTRATION * a_t * np.exp(-0.5 * ((dx * dx) / (sigma_x_t * sigma_x_t) + (dy * dy) / (sigma_y_t * sigma_y_t)))
        return density

    # Create a grid of (x, y) coordinates
    x = np.linspace(0, WIDTH, 128)
    y = np.linspace(0, HEIGHT, 128)
    X, Y = np.meshgrid(x, y)

    # Calculate initial Gaussian density
    Z = calculate_gaussian_density(0, X, Y)
    im = ax.imshow(Z, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='viridis')
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Chemical Concentration')

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
        im.set_data(calculate_gaussian_density(0, X, Y))
        ax.set_title(sub_states_map[sub_states[0][0]])
        return scatters + [im]

    # Animation update function
    def update(frame):
        for i, (scatter, trace) in enumerate(zip(scatters, traces)):
            scatter.set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])

            # Handle trace with periodic boundary conditions
            trace_x = []
            trace_y = []
            start_frame = max(0, frame - 20)  # Limit trace to 20 frames
            for j in range(start_frame, frame + 1):
                x_prev, y_prev = position_matrix[i][j - 1] if j > 0 else position_matrix[i][j]
                x_curr, y_curr = position_matrix[i][j]

                # Check for boundary crossings and adjust coordinates
                if abs(x_curr - x_prev) >= WIDTH / 2 or abs(y_curr - y_prev) >= HEIGHT / 2:
                    break

                trace_x.append(x_curr)
                trace_y.append(y_curr)

            trace.set_data(trace_x, trace_y)

        # Update the Gaussian density function
        Z = calculate_gaussian_density(frame, X, Y)
        im.set_data(Z)
        ax.set_title(sub_states_map[sub_states[0][frame]])
        return scatters + traces + [im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=int(N_STEPS // LOGGING_INTERVAL), blit=False
    )
    anim.save(
        dest_file_path + f"N_{N}_LOGGING_INTERVAL_{LOGGING_INTERVAL}_N_STEPS_{N_STEPS}.mp4",
        writer='ffmpeg', fps=fps
    )


def load_and_animate_agents_and_multiple_heatmaps(json_file_path, primary_heatmap_folder, additional_heatmap_folder, single_file_name1, single_file_name2):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    parameters = data['parameters']
    N = parameters['N']
    WORM_COUNT = parameters['WORM_COUNT']
    LOGGING_INTERVAL = parameters['LOGGING_INTERVAL']
    N_STEPS = parameters['N_STEPS']
    WIDTH = parameters['WIDTH']
    HEIGHT = parameters['HEIGHT']
    MAX_CONCENTRATION = parameters['MAX_CONCENTRATION']
    print(parameters)

    # Prepare the figure and axis
    fig, ax = plt.subplots(1,2, figsize=(16,8))
    ax[0].set_xlim(0, WIDTH)
    ax[0].set_ylim(0, HEIGHT)
    ax[1].set_xlim(0, WIDTH)
    ax[1].set_ylim(0, HEIGHT)    

    # Create a list of scatter plot objects for each agent
    primary_scatters = [ax[0].plot([], [], 'o', color='magenta', markersize=1)[0] for _ in range(WORM_COUNT)]
    additional_scatters = [ax[1].plot([], [], 'o', color='magenta', markersize=1)[0] for _ in range(WORM_COUNT)]
    # position_matrix = [[data[str(agent)][timestep] for timestep in range(int(N_STEPS//LOGGING_INTERVAL))] for agent in range(N)]
    position_matrix = np.array(data['positions'])

    # Parse primary heatmap data from .txt files
    timesteps = N_STEPS // LOGGING_INTERVAL
    primary_matrices = np.zeros((N_STEPS//LOGGING_INTERVAL, N, N))
    for tidx, t in enumerate(range(0, N_STEPS, LOGGING_INTERVAL)):
        file_path = os.path.join(primary_heatmap_folder, f'{single_file_name1}_{t}.txt')
        with open(file_path, 'r') as f:
            matrix = np.transpose(np.loadtxt(f))
            primary_matrices[tidx] = matrix

    # Parse additional heatmap data from .txt files
    additional_matrices = np.zeros((N_STEPS//LOGGING_INTERVAL, N, N))
    for tidx, t in enumerate(range(0, N_STEPS, LOGGING_INTERVAL)):
        file_path = os.path.join(additional_heatmap_folder, f'{single_file_name2}_{t}.txt')
        with open(file_path, 'r') as f:
            matrix = np.transpose(np.loadtxt(f))
            additional_matrices[tidx] = matrix

    primary_grid = primary_matrices.copy()
    additional_grid = additional_matrices.copy()
    primary_im = ax[0].imshow(primary_grid[0], extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Blues', alpha=0.5, vmin=0.0, vmax=MAX_CONCENTRATION)
    additional_im = ax[1].imshow(additional_grid[0], extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Reds', alpha=0.5, vmin=0.0, vmax=MAX_CONCENTRATION)

    # Add colorbars
    primary_cbar = fig.colorbar(primary_im, ax=ax[0])
    primary_cbar.set_label('Attractive Pheromone')
    additional_cbar = fig.colorbar(additional_im, ax=ax[1])
    additional_cbar.set_label('Repulsive Pheromone')

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(zip(primary_scatters, additional_scatters)):
            scatter[0].set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
            scatter[1].set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
        primary_im.set_data(primary_grid[0])
        additional_im.set_data(additional_grid[0])
        return [primary_scatters, additional_scatters] + [primary_im, additional_im]

    # Animation update function
    def update(frame):
        for i, scatter in enumerate(zip(primary_scatters, additional_scatters)):
            scatter[0].set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])
            scatter[1].set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])
        primary_im.set_data(primary_grid[frame])
        additional_im.set_data(additional_grid[frame])
        return [primary_scatters, additional_scatters] + [primary_im, additional_im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save('animation.mp4', writer='ffmpeg', fps=15)


# Main execution
if __name__ == "__main__":
    base_dir = "/home/carlo/babots/cuda_agent_based_sim/json/"
    logs_dir = "/home/carlo/babots/cuda_agent_based_sim/logs/"
    #load_and_animate_agents_and_grid2(base_dir + "agents_all_data.json", fps=30, dest_file_path=base_dir)
    load_and_animate_agents_and_multiple_heatmaps(base_dir + "agents_all_data.json", 
                                                  logs_dir + "attractive_pheromone/", logs_dir + "repulsive_pheromone/", 
                                                  "attractive_pheromone_step", "repulsive_pheromone_step")
