import wandb, re
import matplotlib.pyplot as plt
import numpy as np, math

PREFIX = "gpt-qe/gptqe"
api = wandb.Api()
runs = api.runs(PREFIX)


def fetch(p, key2):
    map = {}
    key1 = 'trainer/global_step'
    for run in runs:
        if p.match(run.name) is not None:
            run = api.run(PREFIX + "/" + run.id)
            history = run.scan_history(keys=[key1, key2])
            for step in history:
                index = step[key1]
                if index not in map:
                    map[index] = []
                map[index].append(step[key2])
    xs = []
    ys = []
    errors = []
    for index, array in sorted(map.items(), key=lambda m: m[0]):
        index = index + 2
        if index % 10 == 0:
            xs.append(index)
            ys.append(np.mean(array))
            errors.append(np.std(array) / math.sqrt(len(array)))
        if index == 500:
            print(array)
    return xs, ys, errors


def plot(title, key, file, realm):
    xs, ys, errors = fetch(re.compile("N23_(?!3047_)\d+_run_1126_*"), key)
    plt.errorbar(xs, ys, errors, color="#008176", linewidth=1, label="with pre-training", marker="o")
    xs, ys, errors = fetch(re.compile("N214_(?!3047_)\d+_run_1124_*"), key)
    plt.errorbar(xs, ys, errors, color='#999999', label="without pre-training", marker="^")
    plt.xlabel('step', fontsize=12)
    plt.ylabel('energy value (Hartree)', fontsize=12)
    plt.plot([0, 505], [-107.591, -107.591], color="#333333")
    plt.xlim([0, 505])
    plt.ylim(realm)
    plt.legend(fontsize=10, loc='upper right')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.title(title)
    plt.savefig(file)
    plt.clf()


if __name__ == '__main__':
    plot("The min energy for each step", 'min_energy at 1.4', "test.pdf", [-107.60, -107.52])
    plot("The mean energy for each step", 'mean energy at 1.4', "test2.pdf", [-107.52, -107.35])
