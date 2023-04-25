import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

if __name__ == '__main__':
    xs = []
    ys = []
    plt.title('2-qubit Heisenberg Hamiltonian')
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('energy')
    with open('../output/two-heisenberg.txt') as f:
        for l in f.readlines():
            id, t = l.rstrip().split('\t')
            en = float(make_tuple(t)[0])
            xs.append(int(id))
            ys.append(en)
    plt.plot(xs, ys, label='Simple model')

    xs = []
    ys = []
    with open('../output/two-heisenberg-deep_energy.tsv') as f:
        for l in f.readlines():
            id, t = l.rstrip().split('\t')
            xs.append(int(id))
            ys.append(float(t))
    plt.plot(xs, ys, label='Deep energy-based model')

    plt.plot([0, 1000], [-6, -6], label='theory')
    plt.legend()
    plt.savefig('output/two.pdf')
