import map_elites.cvt as cvt_map_elites
from typing import List
import subprocess
import logging
import numpy as np

def run_simulation(weights: List[float]):
    num_experiments = 3
    fitness = 0
    msd = 0
    density = 0

    weights[1:4] *= 0.1
    weights[5:8] *= 0.1
    weights[8] *= 7

    for i in range(num_experiments):
        output = subprocess.check_output(['./main', str(weights[0].item()), str(weights[1].item()),
                                           str(weights[2].item()), str(weights[3].item()),  str(weights[4].item()), str(weights[5].item()),
                                            str(weights[6].item()), str(weights[7].item()), str(weights[8].item()), "0"], text=True).split()
        # output = subprocess.check_output(['./main', str(weights[0].item()), "0.0500", "0.0300", "0.0800",
        #                                     str(weights[1].item()), "0.0300", "0.0200", "0.0050", "7", "0"], text=True).split()        
        fitness += int(output[0])
        msd += float(output[1])
        density += float(output[2])
        
    fitness /= num_experiments
    msd /= num_experiments
    density /= num_experiments

    logging.info(f"     Solution = {weights}, fitness = {fitness:.2f}, msd = {msd:.2f}, pheromone density = {density:.2f}")
    return fitness, np.array([msd / 300, density / 30]) # 20 for worm density

if __name__ == "__main__":

    logging.basicConfig(
        filename="./map_elites/res.log",
        encoding="utf-8",
        filemode="w",
        level=logging.INFO
    )

    params = {
        # more of this -> higher-quality CVT
        "cvt_samples": 25000,
        # we evaluate in batches to parallelize
        "batch_size": 15,
        # proportion of niches to be filled before starting
        "random_init": 0.1,
        # batch for random initialization
        "random_init_batch": 100,
        # when to write results (one generation = one batch)
        "dump_period": 10,
        # do we use several cores?
        "parallel": True,
        # do we cache the result of CVT and reuse?
        "cvt_use_cache": True,
        # min/max of parameters
        "min": 0,
        "max": 1,
        # only useful if you use the 'iso_dd' variation operator
        "iso_sigma": 0.01,
        "line_sigma": 0.2        
    }
    archive = cvt_map_elites.compute(2, 9, run_simulation, n_niches=1000, max_evals=1000, log_file=open('./map_elites/cvt.log', 'w'),
                                     params=params, resume=True)
