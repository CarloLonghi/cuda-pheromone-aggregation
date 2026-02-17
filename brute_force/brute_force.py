from typing import List
import subprocess
import numpy as np
import os

def run_simulation(w, folder, num_experiments = 3):
    np.set_printoptions(suppress=True)

    for i in range(num_experiments):
        subprocess.check_output(['./main', "0", "0", "0", "0", "0", "0", "0", "0", "0.0",
                                           np.array2string(w[0]), np.array2string(w[1]),
                                           np.array2string(w[2]),  np.array2string(w[3]),
                                           "1", folder, str(i)], text=True).split()
        print(i)
        # output = subprocess.check_output(['./main', np.array2string(weights[0]), "0.0500", "0.0300", "0.0800",
        #                                     np.array2string(weights[1]), "0.0300", "0.0200", "0.0050", "7", "0"], text=True).split()        

        # desc[i, 0] = float(output[0])
        # desc[i, 1] = float(output[1])
        # desc[i, 2] = float(output[2])

    # return desc

if __name__ == "__main__":

    folder = './bf/'
    file = 'res.npy'
    backup = 'backup.txt'

    num_experiments = 10

    w0v = np.linspace(0, 1.0, 4)
    w1v = np.linspace(0, 0.5, 4)
    w2v = np.linspace(0.0, 0.2, 4)
    w3v = np.linspace(0, 0.5, 4)

    if os.path.isfile(folder + backup):
        with open(folder + backup, "r") as f:
            content = f.read().strip().split(',')
        params = list(map(int, content))
    else:
        params = [0, 0, 0, 0]

    print("Resuming from ", params)

    for id0 in range(params[0], w0v.shape[0]):
        for id1 in range(params[1] if id0 == params[0] else 0, w1v.shape[0]):
            for id2 in range(params[2] if id0 == params[0] and id1 == params[1] else 0, w2v.shape[0]):
                for id3 in range(params[3] if id0 == params[0] and id1 == params[1] and id2 == params[2] else 0, w3v.shape[0]):
                    with open(folder + backup, "w") as f:
                        f.write(f"{id0},{id1},{id2},{id3}")

                    resfolder = "/media/carlo/HD2/res_half/" + str(id0) + str(id1) + str(id2) + str(id3)
                    run_simulation(np.array([w0v[id0], w1v[id1], w2v[id2], w3v[id3]]), resfolder, num_experiments)
                    # results = np.load(folder + file)
                    # results[id0, id1, id2, id3] = r
                    # np.save(folder + file, results)

                    print(id0, id1, id2, id3)
