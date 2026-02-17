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

def animate_ang(pos, ang, elong, cluster, skip_frames):
    # Prepare the figure and axis
    fig, ax = plt.subplots(1,3, figsize=(24,8))
    ax[0].set_xlim(0, WIDTH)
    ax[0].set_ylim(0, HEIGHT)

    # Create a list of scatter plot objects for each agent
    primary_scatters = [ax[0].plot([], [], marker=(1,1,0), color='#BC4749', markersize=12, alpha=1.0)[0] for _ in range(WORM_COUNT)]
    # position_matrix = [[data[str(agent)][timestep] for timestep in range(int(N_STEPS//LOGGING_INTERVAL))] for agent in range(N)]
    position_matrix = pos

    # Initialize empty lines for each replicate
    elong_lines = []
    cluster_lines = []
    for i in range(cluster.shape[0]):
        elong_line, = ax[1].plot([], [], alpha=0.3, color='gray', linewidth=1)
        elong_lines.append(elong_line)
        cluster_line, = ax[2].plot([], [], alpha=0.3, color='gray', linewidth=1)
        cluster_lines.append(cluster_line)        

    # Initialize mean line
    e_mean_line, = ax[1].plot([], [], color='blue', linewidth=2, label='Mean')    
    c_mean_line, = ax[2].plot([], [], color='blue', linewidth=2, label='Mean')   

    t = np.linspace(10, 300, 29) 

    ax[1].set_xlim(0, 300)
    ax[1].set_ylim(0, 4.5)
    ax[1].set_xlabel('Time step')
    ax[1].set_ylabel('Elongation')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)    
    ax[2].set_xlim(0, 300)
    ax[2].set_ylim(0, 1.0)
    ax[2].set_xlabel('Time step')
    ax[2].set_ylabel('Clustering')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)        

    print(f"[0/{int(TIME/DT/LOGGING_INTERVAL)}] frames processed")

    # Parse primary heatmap data from .txt files
    timesteps = int(TIME/DT/LOGGING_INTERVAL / skip_frames) - 1

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(zip(primary_scatters,)):
            scatter[0].set_data([position_matrix[i*2]], [position_matrix[i*2+1]])
            scatter[0].set_marker((1,1,ang[i]))

        for elong_line in elong_lines:
            elong_line.set_data([], [])
        e_mean_line.set_data([], [])
        
        for line in cluster_lines:
            line.set_data([], [])
        c_mean_line.set_data([], [])           

        return [primary_scatters,] + elong_lines + [e_mean_line] + cluster_lines + [c_mean_line]

    # Animation update function
    def update(frame):
        print(f"\033[F[{frame*skip_frames+skip_frames}/{int(TIME/DT/LOGGING_INTERVAL)}] frames processed" )

        for i, scatter in enumerate(zip(primary_scatters,)):
            scatter[0].set_data([position_matrix[(frame*skip_frames*WORM_COUNT+i)*2]], [position_matrix[(frame*skip_frames*WORM_COUNT+i)*2 + 1]])
            scatter[0].set_marker((1,1,ang[frame*skip_frames*WORM_COUNT+i]))
    
        for i, line in enumerate(elong_lines):
            line.set_data(t[:frame], elong[i, :frame])
        e_mean = np.mean(elong[:, :frame], axis=0)
        e_mean_line.set_data(t[:frame], e_mean)
        
        for i, line in enumerate(cluster_lines):
            line.set_data(t[:frame], cluster[i, :frame])
        c_mean = np.mean(cluster[:, :frame], axis=0)
        c_mean_line.set_data(t[:frame], c_mean)    

        return [primary_scatters,] + elong_lines + [e_mean_line] + cluster_lines + [c_mean_line]
            

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save('animation.mp4', writer='ffmpeg', fps=10)


# Main execution
if __name__ == "__main__":

    i = 6
    j = 0
    k = 2
    m = 2    

    base_dir = f"/media/carlo/EXTERNAL_USB/res_res/{str(i)}{str(j)}{str(k)}{str(m)}/"
    logs_dir = "./logs/"
    WIDTH = 128
    HEIGHT = 128
    WORM_COUNT = 10000
    TIME = 300
    DT = 0.1
    LOGGING_INTERVAL = 10
    skip_frames = 10
    # load_and_animate_agents_and_grid2(base_dir + "agents_all_data.json", fps=30, dest_file_path=base_dir)
    # args, pos, angles = load_data_txt(base_dir + "1.txt")
    pos = read_zlib_compressed_floats(base_dir+"0_pos.dat", WORM_COUNT*int(TIME/DT/LOGGING_INTERVAL)*2)
    ang = read_zlib_compressed_floats(base_dir+"0_ang.dat", WORM_COUNT*int(TIME/DT/LOGGING_INTERVAL))
    ang = ang / np.pi * 180 + 90

    elong = np.zeros((10,29)) - 1
    cluster = np.zeros((10,29)) - 1

    for n in range(10):
        file = "./res/res/" + str(i) + str(j) + str(k) + str(m) + "_" + str(n) + "_res.dat"
        if os.path.isfile(file):
            d = read_zlib_compressed_floats(file, 30 * WORM_COUNT + 30 + 1)
            cluster[n] = [np.max(d[t * WORM_COUNT : (t + 1) * WORM_COUNT]) / 10000 for t in range(1, 30)]
            elong[n] = d[30 * WORM_COUNT + 1 : 30 * WORM_COUNT + 30]


    animate_ang(pos, ang, elong, cluster, skip_frames)
