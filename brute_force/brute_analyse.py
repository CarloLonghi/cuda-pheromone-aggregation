from typing import List
import subprocess
import numpy as np
import os
import concurrent.futures
import multiprocessing as mp

def run_analysis(exp_folder, folder, exp_id):
    np.set_printoptions(suppress=True)
    subprocess.check_output(['./analyse', exp_folder, folder, str(exp_id)]).split()

def parallel_run(exp_folder, folder, num_experiments = 5):
    # Use ProcessPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_experiments) as executor: # mp.cpu_count()
        # Submit all tasks
        futures = [executor.submit(run_analysis, exp_folder, folder, i) for i in range(0, num_experiments)]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()    

if __name__ == "__main__":

    backup = './bf/analyse_backup.txt'
    exp_folder = '/media/carlo/HD2/res_half/' 

    num_experiments = 10

    if os.path.isfile(backup):
        with open(backup, "r") as f:
            content = f.read().strip().split(',')
        params = list(map(int, content))
    else:
        params = [0, 0, 0, 0]

    print("Resuming from ", params)

    for id0 in range(params[0], 4):
        for id1 in range(params[1] if id0 == params[0] else 0, 4):
            for id2 in range(params[2] if id0 == params[0] and id1 == params[1] else 0, 4):
                for id3 in range(params[3] if id0 == params[0] and id1 == params[1] and id2 == params[2] else 0, 4):

                    with open(backup, "w") as f:
                        f.write(str(id0)+","+str(id1)+","+str(id2)+","+str(id3))
                                            
                    folder = str(id0) + str(id1) + str(id2) + str(id3)
                    parallel_run(exp_folder, folder, num_experiments)
                

                    print(id0, id1, id2, id3)
