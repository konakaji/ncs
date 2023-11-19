import sys
import os
import numpy as np

DIR = "../output/"
counts = [0, 1]

if __name__ == '__main__':
    molecule_name = sys.argv[1]
    m = {}
    for count in counts:
        file = f"{molecule_name}_{count}.txt"
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
            f.write(f"{dist}\t{np.mean(values)}\n")
