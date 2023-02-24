import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

if __name__ == '__main__':
    xs = []
    ys = []
    with open('output/two-heisenberg.txt') as f:
        for l in f.readlines():
            id, t = l.rstrip().split('\t')
            en = float(make_tuple(t)[0])
            xs.append(int(id))
            ys.append(en)
    plt.title('GQE with qDRIFT based model.')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('energy')
    plt.plot(xs, ys, label='GQE')
    plt.plot([0, 100], [-6, -6], label='theory')
    plt.legend()
    plt.show()
