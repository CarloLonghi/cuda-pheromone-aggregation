import os
import numpy as np
from matplotlib import pyplot as plt
import zlib
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

TIME = 300
DT = 0.1
N_STEPS = int(TIME / DT)
LOGGING_INTERVAL = 10
WORM_COUNT =  10000

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


# Read the data

timesteps = 300

cluster = np.zeros((8,8,8,8,10)) - 1
elong = np.zeros((8,8,8,8,10)) - 1
msd = np.zeros((8,8,8,8,10)) - 1

reference_msd = 0

for i in range(8):
    for j in range(8):
        for k in range(8):
            for m in range(8):
                for n in range(10):
                    file = "./res/res/" + str(i) + str(j) + str(k) + str(m) + "_" + str(n) + "_res.dat"
                    if os.path.isfile(file):
                        d = read_zlib_compressed_floats(file, 30 * WORM_COUNT + 30 + 1)
                        # cluster[i,j,k,m,n] = np.max(d[29 * WORM_COUNT : 30 * WORM_COUNT], axis=-1)
                        cluster[i,j,k,m,n] = np.mean([np.max(d[t * WORM_COUNT : (t + 1) * WORM_COUNT]) / 10000 for t in range(30 - 1)])
                        # cluster[i,j,k,m,n] = np.mean([1 / np.sum(d[t * WORM_COUNT : (t + 1) * WORM_COUNT] ** 2) for t in range(30 - 1)])
                        # elong[i,j,k,m,n] = d[30 * WORM_COUNT + 30 - 1]
                        elong[i,j,k,m,n] = np.mean(d[30 * WORM_COUNT : 30 * WORM_COUNT + 30])
                        msd[i,j,k,m,n] = d[-1]

c = np.mean(cluster, axis=(-1))
e = np.mean(elong, axis=(-1))
m = np.mean(msd, axis=(-1))

m /= m[0,0,0,0]

results = np.zeros((8, 8, 8, 8, 3))
results[..., 0] = c
results[..., 1] = e
results[..., 2] = m
np.save("./res/final.npy", results)

# K-means clustering
points = np.swapaxes(np.array([c.flatten(), m.flatten(), e.flatten()]), 0, 1)

# Standardize the data (important for distance-based methods)
scaler = StandardScaler()
points_scaled = scaler.fit_transform(points)

# Option 1: DBSCAN (automatic cluster detection)
# clustering = DBSCAN(eps=0.5, min_samples=5).fit(points_scaled)
# labels = clustering.labels_

# Option 2: K-Means (if you know k)
kmeans = KMeans(n_clusters=4).fit(points_scaled)
labels = kmeans.labels_

# Plot the data
fig = plt.figure(figsize=(10,10))
fig.tight_layout()
ax = fig.subplots(1, 1, subplot_kw=dict(projection='3d'))

ax.scatter(c.flatten(), m.flatten(), e.flatten(), c=labels)
ax.scatter(c.flatten()[0], m.flatten()[0], e.flatten()[0], c="red", marker="*")
ax.set_ylabel("MSD", fontsize=15)
ax.set_xlabel("Clustering", fontsize=15)
ax.set_zlabel("Elongation", fontsize=15)
plt.tight_layout()
fig.savefig('./poster/temp.png', transparent=True)
plt.show()