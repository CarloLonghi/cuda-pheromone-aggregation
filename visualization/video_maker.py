import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
import zlib
import struct

def read_zlib_compressed_floats(filename, expected_size=None):
    """
    Read zlib-compressed float array written by C++
    
    Args:
        filename: Path to the compressed file
        expected_size: Number of floats expected (optional)
    
    Returns:
        numpy array of floats
    """
    # Read the compressed data
    with open(filename, 'rb') as f:
        compressed_data = f.read()
    
    # Decompress the data
    try:
        # Get decompressed size if known
        if expected_size is not None:
            decompressed_size = expected_size * 4  # 4 bytes per float
            decompressed_data = zlib.decompress(compressed_data)
        else:
            decompressed_data = zlib.decompress(compressed_data)
            
    except zlib.error as e:
        print(f"Decompression error: {e}")
        return None
    
    # Convert bytes to float array
    num_floats = len(decompressed_data) // 4
    float_array = np.frombuffer(decompressed_data, dtype=np.float32, count=num_floats)
    
    return float_array

def load_data_txt(file):
    with open(file, "r") as f:
        lines = f.readlines()
        data = lines[0].split()

    WIDTH = float(data[0])
    HEIGHT = float(data[1])
    N = int(data[2]) 
    WORM_COUNT = int(data[3])
    TIME = int(data[4])
    
    args = [WIDTH, HEIGHT, N, WORM_COUNT, TIME]

    data = data[5:]

    position_matrix = np.zeros((TIME, WORM_COUNT, 2))
    angles = np.zeros((TIME, WORM_COUNT))
    for i in range(TIME):
        for j in range(WORM_COUNT):
            position_matrix[i, j, 0] = float(data[(i * WORM_COUNT + j) * 2])
            position_matrix[i, j, 1] = float(data[(i * WORM_COUNT + j) * 2 + 1])
            angles[i, j] = float(data[TIME * WORM_COUNT * 2 + i * WORM_COUNT + j])

    position_matrix = np.swapaxes(position_matrix, 0, 1)
            
    return args, position_matrix, angles


def load_data_json(json_file_path):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    parameters = data['parameters']
    N = parameters['N']
    WORM_COUNT = parameters['WORM_COUNT']
    TIME = parameters['TIME']
    WIDTH = parameters['WIDTH']
    HEIGHT = parameters['HEIGHT']
    print(parameters)

    args = [WIDTH, HEIGHT, N, WORM_COUNT, TIME]
    pos = data['positions']
    angles = data['angles']
    
    return args, pos, angles

def animate(pos, primary_heatmap_folder, additional_heatmap_folder, single_file_name1, single_file_name2):

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
    position_matrix = pos

    print(f"[0/{int(TIME/DT/LOGGING_INTERVAL)}] frames processed")

    # Parse primary heatmap data from .txt files
    timesteps = int(TIME / 2)
    # primary_frame = np.zeros((N, N))
    # file_path = os.path.join(primary_heatmap_folder, f'{single_file_name1}_{0}.txt')
    # with open(file_path, 'r') as f:
    #     matrix = np.transpose(np.loadtxt(f))
    #     primary_frame = matrix

    # # Parse additional heatmap data from .txt files
    # additional_frame = np.zeros((N, N))
    # file_path = os.path.join(additional_heatmap_folder, f'{single_file_name2}_{0}.txt')
    # with open(file_path, 'r') as f:
    #     matrix = np.transpose(np.loadtxt(f))
    #     additional_frame = matrix

    # primary_im = ax[0].imshow(primary_frame, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Blues', alpha=0.5, vmin=0.0, vmax=primary_frame.max())
    # additional_im = ax[1].imshow(additional_frame, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Reds', alpha=0.5, vmin=0.0, vmax=additional_frame.max())

    # # Add colorbars
    # primary_cbar = fig.colorbar(primary_im, ax=ax[0])
    # primary_cbar.set_label('Attractive Pheromone')
    # additional_cbar = fig.colorbar(additional_im, ax=ax[1])
    # additional_cbar.set_label('Repulsive Pheromone')

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(zip(primary_scatters, additional_scatters)):
            # scatter[0].set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
            # scatter[1].set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
            scatter[0].set_data([position_matrix[i*2]], [position_matrix[i*2+1]])
            scatter[1].set_data([position_matrix[i*2]], [position_matrix[i*2+1]])            
        # primary_im.set_data(primary_frame)
        # additional_im.set_data(additional_frame)
        return [primary_scatters, additional_scatters] # + [primary_im, additional_im]

    # Animation update function
    def update(frame):
        # primary_frame = np.zeros((N, N))
        # additional_frame = np.zeros((N, N))
        # file_path = os.path.join(primary_heatmap_folder, f'{single_file_name1}_{frame}.txt')
        # with open(file_path, 'r') as f:
        #     matrix = np.transpose(np.loadtxt(f))
        #     primary_frame = matrix        

        # file_path = os.path.join(additional_heatmap_folder, f'{single_file_name2}_{frame}.txt')
        # with open(file_path, 'r') as f:
        #     matrix = np.transpose(np.loadtxt(f))
        #     additional_frame = matrix
        print(f"\033[F[{frame*20+20}/{int(TIME/DT/LOGGING_INTERVAL)}] frames processed" )

        for i, scatter in enumerate(zip(primary_scatters, additional_scatters)):
            # scatter[0].set_data([position_matrix[i][frame*10][0]], [position_matrix[i][frame*10][1]])
            # scatter[1].set_data([position_matrix[i][frame*10][0]], [position_matrix[i][frame*10][1]])
            scatter[0].set_data([position_matrix[(frame*20*WORM_COUNT+i)*2]], [position_matrix[(frame*20*WORM_COUNT+i)*2 + 1]])
            scatter[1].set_data([position_matrix[(frame*20*WORM_COUNT+i)*2]], [position_matrix[(frame*20*WORM_COUNT+i)*2 + 1]])            
        # primary_im.set_data(primary_frame)
        # additional_im.set_data(additional_frame)
        return [primary_scatters, additional_scatters] # + [primary_im, additional_im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save('animation.mp4', writer='ffmpeg', fps=1)


def animate_ang(pos, ang, primary_heatmap_folder, additional_heatmap_folder, single_file_name1, single_file_name2):

    # Prepare the figure and axis
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)

    # Create a list of scatter plot objects for each agent
    primary_scatters = [ax.plot([], [], marker=(1,1,0), color='magenta', markersize=6)[0] for _ in range(WORM_COUNT)]
    # position_matrix = [[data[str(agent)][timestep] for timestep in range(int(N_STEPS//LOGGING_INTERVAL))] for agent in range(N)]
    position_matrix = pos

    print(f"[0/{int(TIME/DT/LOGGING_INTERVAL)}] frames processed")

    # Parse primary heatmap data from .txt files
    timesteps = int(TIME / 10)

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(zip(primary_scatters,)):
            scatter[0].set_data([position_matrix[i*2]], [position_matrix[i*2+1]])
            scatter[0].set_marker((1,1,ang[i]))
        return [primary_scatters,] # + [primary_im, additional_im]

    # Animation update function
    def update(frame):
        print(f"\033[F[{frame*10+10}/{int(TIME/DT/LOGGING_INTERVAL)}] frames processed" )

        for i, scatter in enumerate(zip(primary_scatters,)):
            scatter[0].set_data([position_matrix[(frame*10*WORM_COUNT+i)*2]], [position_matrix[(frame*10*WORM_COUNT+i)*2 + 1]])
            scatter[0].set_marker((1,1,ang[frame*2*WORM_COUNT+i]))
        return [primary_scatters,] # + [primary_im, additional_im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save('animation.mp4', writer='ffmpeg', fps=1)


# Main execution
if __name__ == "__main__":
    base_dir = "./json/"
    logs_dir = "./logs/"
    WIDTH = 50
    HEIGHT = 50
    WORM_COUNT = 10000
    TIME = 900   
    DT = 0.1
    LOGGING_INTERVAL = 10
    # load_and_animate_agents_and_grid2(base_dir + "agents_all_data.json", fps=30, dest_file_path=base_dir)
    # args, pos, angles = load_data_txt(base_dir + "1.txt")
    pos = read_zlib_compressed_floats(base_dir+"0_pos.dat", WORM_COUNT*int(TIME/DT/LOGGING_INTERVAL)*2)
    ang = read_zlib_compressed_floats(base_dir+"0_ang.dat", WORM_COUNT*int(TIME/DT/LOGGING_INTERVAL))
    ang = ang / np.pi * 180 + 90
    animate_ang(pos, ang, 
            logs_dir + "attractive_pheromone/", logs_dir + "repulsive_pheromone/", 
            "attractive_pheromone_step", "repulsive_pheromone_step")
