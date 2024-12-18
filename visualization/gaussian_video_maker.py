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
        return ATTRACTION_STRENGTH * np.log(density + ATTRACTION_SCALE)

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

# Main execution
if __name__ == "__main__":
    base_dir = "/path/to/your/json/dir/"
    load_and_animate_agents_and_grid2(base_dir + "agents_all_data.json", fps=30, dest_file_path=base_dir)
