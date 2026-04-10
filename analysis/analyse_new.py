import numpy as np
from scipy.ndimage import gaussian_filter
import zlib
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import center_of_mass
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from skimage.morphology import remove_small_holes

def get_pattern_crop(density, crop_fraction=0.35):
    """Crop around the actual center of mass of the pattern."""
    cy, cx = center_of_mass(density)
    cy, cx = int(cy), int(cx)
    dim = int(min(density.shape) * crop_fraction / 2)
    # Clamp to grid bounds
    y0 = max(0, cy - dim)
    y1 = min(density.shape[0], cy + dim)
    x0 = max(0, cx - dim)
    x1 = min(density.shape[1], cx + dim)
    return density[y0:y1, x0:x1]

def minkowski_functionals(density, threshold=None, binary_override=None,
                          min_size_fraction=0.0003, hole_size_fraction=0.0003):
    """
    min_size_fraction and hole_size_fraction are expressed as a fraction
    of total grid area, so they scale automatically with crop size.
    """
    total_pixels = density.shape[0] * density.shape[1]
    min_size = max(10, int(total_pixels * min_size_fraction))
    hole_size = max(10, int(total_pixels * hole_size_fraction))

    if binary_override is not None:
        binary = binary_override
    else:
        if threshold is None:
            threshold = threshold_otsu(density)
        binary = density > threshold

    binary = remove_small_objects(binary, min_size=min_size)
    binary = remove_small_holes(binary, area_threshold=hole_size)

    # plt.imshow(binary)
    # plt.show()

    # W0: Area fraction
    W0 = binary.mean()

    # W1: Perimeter
    eroded = binary_erosion(binary)
    perimeter_pixels = binary & ~eroded
    W1 = perimeter_pixels.sum() / binary.size

    # W2: Euler characteristic
    labeled_fg = label(binary, connectivity=1)
    n_objects = labeled_fg.max()
    labeled_bg = label(~binary, connectivity=1)
    n_holes = labeled_bg.max() - 1
    W2 = n_objects - n_holes

    compactness = W1 / (W0 + 1e-12)

    return {"W0_area": W0, "W1_perimeter": W1, "W2_euler": W2, "compactness": compactness}

def minkowski_curves(density, n_thresholds=50):
    """Sweep thresholds to get Minkowski functional curves."""
    thresholds = np.linspace(density.min(), density.max(), n_thresholds)
    results = [minkowski_functionals(density, t) for t in thresholds]
    return thresholds, results

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

def radial_power_spectrum(density, high_pass_fraction=0.15):
    N = density.shape[0]
    high_pass_sigma = N * high_pass_fraction  # scales with crop size
    trend = gaussian_filter(density, sigma=high_pass_sigma)
    field = density - trend

    window = np.outer(np.hanning(N), np.hanning(N))
    F = np.fft.fft2(field * window)
    power = np.abs(np.fft.fftshift(F))**2

    cx, cy = N // 2, N // 2
    y_idx, x_idx = np.indices((N, N))
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2).astype(int)
    radial_mean = np.bincount(r.ravel(), weights=power.ravel())
    radial_count = np.bincount(r.ravel())
    P = radial_mean / np.maximum(radial_count, 1)

    k_vals = np.arange(len(P)) * (2 * np.pi / N)
    return k_vals, P

def agents_to_density(positions, coord_max=128.0, grid_size=256, sigma=None):
    density = np.zeros((grid_size, grid_size))
    x = ((positions[:, 0] / coord_max) * grid_size).astype(int).clip(0, grid_size - 1)
    y = ((positions[:, 1] / coord_max) * grid_size).astype(int).clip(0, grid_size - 1)
    np.add.at(density, (x, y), 1)
    if sigma is not None:
        density = gaussian_filter(density, sigma=sigma)
    return density        

def radial_density_profile(density, n_bins=50):
    """
    Compute mean density as a function of distance from center of mass.
    
    Parameters
    ----------
    density : 2D numpy array
    n_bins : int
        Number of radial bins.
    
    Returns
    -------
    dict with scalar summary metrics and the full profile arrays for plotting.
    """
    from scipy.ndimage import center_of_mass

    H, W = density.shape
    cy, cx = center_of_mass(density)

    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)

    # Use the largest inscribed circle to avoid edge artifacts
    max_r = min(cx, cy, W - cx, H - cy)
    bins = np.linspace(0, max_r, n_bins + 1)
    r_centers = (bins[:-1] + bins[1:]) / 2

    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if mask.sum() > 0:
            profile[i] = density[mask].mean()

    # Normalize profile to [0, 1] for scale-invariant comparison
    profile_norm = profile / (profile.max() + 1e-12)

    # --- Scalar summary metrics ---

    # 1. Radius containing 50% and 90% of total mass
    # (use profile * shell_area as mass proxy)
    shell_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
    mass_per_bin = profile * shell_areas
    cumulative_mass = np.cumsum(mass_per_bin)
    cumulative_mass /= cumulative_mass[-1] + 1e-12

    r50 = r_centers[np.searchsorted(cumulative_mass, 0.50)]
    r90 = r_centers[np.searchsorted(cumulative_mass, 0.90)]

    # 2. Concentration index: how much mass is in the inner 25% of radius
    # High = compact core (branching), Low = uniform (labyrinth/spots)
    r25_idx = np.searchsorted(r_centers, max_r * 0.25)
    concentration = cumulative_mass[r25_idx] if r25_idx < n_bins else 1.0

    # 3. Profile slope: fit a line to the normalized profile in log-log space
    # Steeper negative slope = faster decay = more compact pattern
    valid = profile_norm > 0.01  # avoid log(0)
    if valid.sum() > 5:
        slope, _ = np.polyfit(np.log(r_centers[valid] + 1),
                               np.log(profile_norm[valid] + 1e-12), 1)
    else:
        slope = 0.0

    # 4. Core-to-periphery ratio: mean density in inner 25% vs outer 25%
    inner = profile[:n_bins // 4].mean()
    outer = max(profile[3 * n_bins // 4:].mean(), profile.max() * 0.01)
    core_periphery_ratio = inner / outer

    return {
        "r50": r50,
        "r90": r90,
        "concentration": concentration,
        "profile_slope": slope,
        "core_periphery_ratio": core_periphery_ratio,
        # Full arrays for plotting — not used as metrics but useful for debugging
        "_profile": profile_norm,
        "_r_centers": r_centers,
    }

def spectral_metrics(k_vals, P, k_min=2):
    P_trim = P[k_min:]
    k_trim = k_vals[k_min:]

    k_star_idx = np.argmax(P_trim)
    k_star = k_trim[k_star_idx]
    peak_height = P_trim[k_star_idx]
    total_power = P_trim.sum()

    # Fraction of total power in the peak (more robust than ratio to median)
    peak_fraction = peak_height / (total_power + 1e-12)

    # Or: ratio of peak to mean of non-peak region
    mask = np.ones(len(P_trim), dtype=bool)
    mask[max(0, k_star_idx-2):k_star_idx+3] = False  # exclude peak neighborhood
    background_mean = P_trim[mask].mean()
    snr = peak_height / (background_mean + 1e-12)

    P_norm = P_trim / (total_power + 1e-12)
    spectral_entropy = -np.sum(P_norm * np.log(P_norm + 1e-12))

    return {
        "k_star": k_star,
        "peak_fraction": peak_fraction,  # 0–1, higher = more dominant peak
        "snr": snr,                       # cleaner sharpness metric
        "spectral_entropy": spectral_entropy
    }

def get_adaptive_crop(density, mass_fraction=0.95, min_fraction=0.2, max_fraction=0.95):
    """
    Crop around the center of mass, using a radius that captures
    `mass_fraction` of the total density mass.
    
    Parameters
    ----------
    mass_fraction : float
        Fraction of total mass to capture (e.g. 0.95 = 95% of agents).
    min_fraction : float
        Minimum crop as fraction of grid size (avoids over-zooming on tiny patterns).
    max_fraction : float
        Maximum crop as fraction of grid size (avoids capturing too much empty space).
    """
    from scipy.ndimage import center_of_mass

    H, W = density.shape
    cy, cx = center_of_mass(density)
    cy, cx = int(cy), int(cx)

    # Compute radial distance from center of mass for every pixel
    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)

    # Find radius that captures `mass_fraction` of total mass
    total_mass = density.sum()
    # Sort pixels by distance and accumulate mass
    order = np.argsort(r.ravel())
    cumulative_mass = np.cumsum(density.ravel()[order])
    cutoff_idx = np.searchsorted(cumulative_mass, mass_fraction * total_mass)
    r_capture = r.ravel()[order[cutoff_idx]]

    # Clamp to min/max bounds
    min_dim = int(min(H, W) * min_fraction / 2)
    max_dim = int(min(H, W) * max_fraction / 2)
    dim = int(np.clip(r_capture, min_dim, max_dim))

    # Crop (clamped to grid bounds)
    y0 = max(0, cy - dim)
    y1 = min(H, cy + dim)
    x0 = max(0, cx - dim)
    x1 = min(W, cx + dim)

    crop = density[y0:y1, x0:x1]
    return crop, dim, (cy, cx)

def analyze_pattern(agent_positions, grid_size=2048, mass_fraction=0.95,
                    min_object_size=20):
    # 1. Bin agents to density without smoothing
    density_full = agents_to_density(agent_positions, grid_size=grid_size, sigma=None)

    # 2. Adaptive crop
    density_crop, crop_dim, center = get_adaptive_crop(
        density_full,
        mass_fraction=mass_fraction,
        min_fraction=0.2,
        max_fraction=0.95
    )

    # 3. Smooth after cropping, sigma scaled to crop size
    crop_size = density_crop.shape[0]
    sigma = crop_size * 0.008
    density = gaussian_filter(density_crop, sigma=sigma)

    # plt.imshow(density)
    # plt.show()

    # 4. Spectral analysis
    k, P = radial_power_spectrum(density, high_pass_fraction=0.15)
    spectral = spectral_metrics(k, P)

    # 5. Minkowski
    mink = minkowski_functionals(density)

    # 6. Radial profile
    radial = radial_density_profile(density)

    # Strip private arrays before returning (keep only scalar metrics)
    radial_scalars = {k: v for k, v in radial.items() if not k.startswith("_")}

    return {
        **spectral,
        **mink,
        "spatial_extent": crop_dim / (grid_size / 2),
        **radial_scalars
    }

def process_single_n(args):
    n, exp_folder, idx = args
    pos_file = f"{exp_folder}{idx}/{n}_pos.dat"
    p = read_zlib_compressed_floats(pos_file, 2 * 10000 * 300)
    
    res = np.zeros((30, 5))
    for t in range(30):
        p_temp = p[(t * 10 * 10000) * 2 : ((t * 10 + 1) * 10000) * 2]
        
        # Vectorized reshape instead of inner loop
        pos = np.array(p_temp).reshape(10000, 2)
        
        r = analyze_pattern(pos, grid_size=2048)
        res[t] = [r["W2_euler"], r["r50"], r["r90"], r["concentration"], r["core_periphery_ratio"]]
    
    return n, res

if __name__ == "__main__":

    # RUN ANALYSIS ON ONE FULL SIMULATION

    # idx = "0002"
    # pos_file = f"/media/carlo/EXTERNAL_USB/res_res/{idx}/0_pos.dat"
    # p = read_zlib_compressed_floats(pos_file, 2 * 10000 * 300)

    # res = np.zeros((30, 4))
    # for t in range(0, 30, 1):
    #     p_temp = p[(t * 10 * 10000) * 2 : ((t * 10 + 1) * 10000) * 2]
    #     pos = np.zeros((10000, 2))
    #     for i in range(10000):
    #         pos[i, 0] = p_temp[i * 2]
    #         pos[i, 1] = p_temp[i * 2 + 1]

    #     r = analyze_pattern(pos, grid_size=2048)
    #     res[t] = [r["W2_euler"], r["spatial_extent"], r["r50"], r["r90"]]

    # plt.plot(res[:, 0])    
    # plt.show()

    # np.save(f"./poster/{idx}_metrics.npy", res)
    

    # RUN ANALYSIS ON A SINGLE TIMESTEP OF A SIMULATION

    # idx = "0002"
    # pos_file = f"/media/carlo/EXTERNAL_USB/res_res/{idx}/0_pos.dat"
    # p = read_zlib_compressed_floats(pos_file, 2 * 10000 * 300)
    # t = 299
    # p_temp = p[(t * 10000) * 2 : ((t + 1) * 10000) * 2]
    # pos = np.zeros((10000, 2))
    # for i in range(10000):
    #     pos[i, 0] = p_temp[i * 2]
    #     pos[i, 1] = p_temp[i * 2 + 1]

    # r = analyze_pattern(pos, grid_size=2048)
    # res = [r["W2_euler"], r["spatial_extent"], r["r50"], r["r90"]]
    # print(res)    


    # RUN ANALYSIS ON FULL DATASET WITH PARALLELIZATION

    backup = './bf/analyse_backup.txt'
    exp_folder = '/media/carlo/EXTERNAL_USB/res_res/'
    res_folder =  '/media/carlo/HD2/res/res'

    num_experiments = 10

    if os.path.isfile(backup):
        with open(backup, "r") as f:
            content = f.read().strip().split(',')
        params = list(map(int, content))
    else:
        params = [0, 0, 0, 0]

    print("Resuming from ", params)

    for id0 in range(params[0], 8):
        for id1 in range(params[1] if id0 == params[0] else 0, 8):
            for id2 in range(params[2] if id0 == params[0] and id1 == params[1] else 0, 8):
                for id3 in range(params[3] if id0 == params[0] and id1 == params[1] and id2 == params[2] else 0, 8):  
                    idx = str(id0) + str(id1) + str(id2) + str(id3)
                    args_list = [(n, exp_folder, idx) for n in range(10)]

                    with ProcessPoolExecutor(max_workers=10) as executor:
                        futures = {executor.submit(process_single_n, args): args[0] for args in args_list}
                        for future in as_completed(futures):
                            n, res = future.result()
                            np.save(f"{res_folder}/{idx}_{n}_new", res)    

                    with open(backup, "w") as f:
                        f.write(str(id0)+","+str(id1)+","+str(id2)+","+str(id3))

                    print(idx)
