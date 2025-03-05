import torch
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import json

# solutions = torch.load('results/solutions.pt')
# solutions[:, 2:] *= 0.01

# idx = 9

# weights = solutions[idx]

weights = torch.tensor([0.8836, 0.9050, 0.3841, 0.9648, 0.3937, 0.6005, 0.2602, 0.7983])
# weights[1:4] *= 0.01
# weights[5:8] *= 0.01
#weights = torch.tensor([1, 0.009, 0.002, 0.0002, 0.0, 0.0, 0.0, 0.0])
weights = torch.tensor([1.0000, 0.0900, 0.0005, 0.0020, 0.0000, 0.0000, 0.0000, 0.0000])

output = subprocess.check_output(['./main', str(weights[0].item()), str(weights[1].item()),
                                           str(weights[2].item()), str(weights[3].item()),  str(weights[4].item()), str(weights[5].item()),
                                            str(weights[6].item()), str(weights[7].item()), "1"], text=True).split()

base_dir = "/home/carlo/babots/cuda_agent_based_sim/json/"
with open(base_dir + "agents_all_data.json", 'r') as f:
    data = json.load(f)

position_matrix = np.array(data['positions'])

def get_adj_matrix(pos_matrix, dist) -> np.ndarray:
    nw = pos_matrix.shape[0]
    adj_matrix = np.zeros((pos_matrix.shape[1], nw, nw))
    for t in range(pos_matrix.shape[1]):
        worm_pos = pos_matrix[:,t]
        dist_mat = np.zeros((nw, nw))
        for w in range(len(worm_pos)):
            dist_mat[w] = np.sqrt((worm_pos[w, 0] - worm_pos[:, 0]) ** 2 + (worm_pos[w, 1] - worm_pos[:, 1]) ** 2)
        adj_matrix[t][dist_mat <= dist] = 1
    return adj_matrix

def find_clusters(graph, interval):
    to_visit = set(range(len(graph)))
    clusters = []
    while len(to_visit) > 0:
        node = to_visit.pop()
        bfs_list = set([node,])
        cluster = set([node,])
        while len(bfs_list) > 0:
            bfs_node = bfs_list.pop()
            if bfs_node in to_visit:
                to_visit.remove(bfs_node)
            connected = [i for i in range(len(graph[bfs_node])) if graph[bfs_node][i] >= interval]
            connected = [c for c in connected if c in to_visit]
            bfs_list.update(connected)
            cluster.update(connected)
        clusters.append(cluster)
    return clusters

am = get_adj_matrix(position_matrix, 1)

max_clusters = np.zeros(am.shape[0])
mean_clusters = np.zeros(am.shape[0])
for t in range(am.shape[0]):
    clusters = find_clusters(am[t], 1)
    cluster_size = [len(c) for c in clusters]
    max_clusters[t] = np.max(cluster_size)
    mean_clusters[t] = np.mean(cluster_size)

colors = ['blue', "orange"]
fig, ax1 = plt.subplots()
ax1.plot(max_clusters, label="Max", color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])
ax1.set_ylabel("Max", color=colors[0])
ax1.set_ylim(1, 10)

ax2 = ax1.twinx()
ax2.plot(mean_clusters, label="Mean", color=colors[1])
ax2.tick_params(axis='y', labelcolor=colors[1])
ax2.set_ylabel("Mean", color=colors[1])
ax2.set_ylim(1, 1.5)

plt.title(f"{weights[0].item():.4f} {weights[1].item():.4f} {weights[2].item():.4f} {weights[3].item():.4f} {weights[4].item():.4f} {weights[5].item():.4f} {weights[6].item():.4f} {weights[7].item():.4f}")

plt.savefig(f"./results/plots/{weights[0].item():.4f}_{weights[1].item():.4f}_{weights[2].item():.4f}_{weights[3].item():.4f}_{weights[4].item():.4f}_{weights[5].item():.4f}_{weights[6].item():.4f}_{weights[7].item():.4f}_test.png")
# plt.show()
