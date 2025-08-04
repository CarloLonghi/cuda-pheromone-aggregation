import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

def load_and_animate_agents_and_multiple_heatmaps(json_file_path, primary_heatmap_folder, additional_heatmap_folder, single_file_name1, single_file_name2):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    parameters = data['parameters']
    N = parameters['N']
    WORM_COUNT = parameters['WORM_COUNT']
    LOGGING_INTERVAL = parameters['LOGGING_INTERVAL']
    TIME = parameters['TIME']
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
    timesteps = TIME
    primary_frame = np.zeros((N, N))
    file_path = os.path.join(primary_heatmap_folder, f'{single_file_name1}_{0}.txt')
    with open(file_path, 'r') as f:
        matrix = np.transpose(np.loadtxt(f))
        primary_frame = matrix

    # Parse additional heatmap data from .txt files
    additional_frame = np.zeros((N, N))
    file_path = os.path.join(additional_heatmap_folder, f'{single_file_name2}_{0}.txt')
    with open(file_path, 'r') as f:
        matrix = np.transpose(np.loadtxt(f))
        additional_frame = matrix

    primary_im = ax[0].imshow(primary_frame, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Blues', alpha=0.5, vmin=0.0, vmax=primary_frame.max())
    additional_im = ax[1].imshow(additional_frame, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Reds', alpha=0.5, vmin=0.0, vmax=additional_frame.max())

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
        primary_im.set_data(primary_frame)
        additional_im.set_data(additional_frame)
        return [primary_scatters, additional_scatters] + [primary_im, additional_im]

    # Animation update function
    def update(frame):
        primary_frame = np.zeros((N, N))
        additional_frame = np.zeros((N, N))
        file_path = os.path.join(primary_heatmap_folder, f'{single_file_name1}_{frame}.txt')
        with open(file_path, 'r') as f:
            matrix = np.transpose(np.loadtxt(f))
            primary_frame = matrix        

        file_path = os.path.join(additional_heatmap_folder, f'{single_file_name2}_{frame}.txt')
        with open(file_path, 'r') as f:
            matrix = np.transpose(np.loadtxt(f))
            additional_frame = matrix

        for i, scatter in enumerate(zip(primary_scatters, additional_scatters)):
            scatter[0].set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])
            scatter[1].set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])
        primary_im.set_data(primary_frame)
        additional_im.set_data(additional_frame)
        return [primary_scatters, additional_scatters] + [primary_im, additional_im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save('animation.mp4', writer='ffmpeg', fps=1)


# Main execution
if __name__ == "__main__":
    base_dir = "./json/"
    logs_dir = "./logs/"
    #load_and_animate_agents_and_grid2(base_dir + "agents_all_data.json", fps=30, dest_file_path=base_dir)
    load_and_animate_agents_and_multiple_heatmaps(base_dir + "agents_all_data.json", 
                                                  logs_dir + "attractive_pheromone/", logs_dir + "repulsive_pheromone/", 
                                                  "attractive_pheromone_step", "repulsive_pheromone_step")
