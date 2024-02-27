import math
import sys
import numpy as np

DIR = "../output/"
counts = [1, 2, 3]

if __name__ == '__main__':
    molecule_name = sys.argv[1]
    seeds_string = sys.argv[2]
    seeds = seeds_string.split(",")
    m = {}
    for seed in seeds:
        file = f"{molecule_name}_{seed}.txt"
        m_name = file.split("_")
        path = DIR + file
        with open(path) as f:
            for l in f.readlines():
                dist, energy = l.rstrip().split("\t")
                dist = float(dist)
                energy = float(energy)
                if dist not in m:
                    m[dist] = []
                m[dist].append(energy)
    with open(DIR + f"{molecule_name}_min.txt", "w") as f:
        for dist, values in m.items():
            f.write(f"{dist}\t{min(values)}\n")
    with open(DIR + f"{molecule_name}_avg.txt", "w") as f:
        for dist, values in m.items():
            f.write(f"{dist}\t{np.mean(values)}\t{np.std(values) / math.sqrt(len(counts))}\n")
