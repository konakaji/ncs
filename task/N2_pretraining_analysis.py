PRE_TRAINING_FORMAT = "../output/N2_pretraining_{}_energy_{}.csv"
TRAINING_FORMAT = "../output/N2_training_{}_energy_{}.csv"

import matplotlib.pyplot as plt

PRETRAINING_COLOR = ["#008176", "#AADDDD"]
TRAINING_COLOR = ["#333333", "#AAAAAA"]


def get_data(file):
    with open(file) as f:
        xs = []
        ys = []
        for j, l in enumerate(f.readlines()):
            items = l.rstrip().split(",")
            xs.append(j)
            ys.append(float(items[1].replace("\"", "")))
    return xs, ys


def plot(category):
    plt.xlabel("step", fontsize=12)
    plt.ylabel("energy (Hartree)", fontsize=12)
    xs, ys = get_data(PRE_TRAINING_FORMAT.format(category, 1))
    plt.plot(xs, ys, linewidth=2, color=PRETRAINING_COLOR[0], label="With pretraining (Best)")

    # for index in [2]:
    #     xs, ys = get_data(PRE_TRAINING_FORMAT.format(category, index))
    #     plt.plot(xs, ys, linewidth=2, linestyle="dotted", color=PRETRAINING_COLOR[1], label="With pretraining (2nd Best)")

    xs, ys = get_data(TRAINING_FORMAT.format(category, 1))
    plt.plot(xs, ys, linewidth=1, linestyle="--", color=TRAINING_COLOR[0], label="Without pretraining (Best)")

    # for index in [2]:
    #     xs, ys = get_data(TRAINING_FORMAT.format(category, index))
    #     plt.plot(xs, ys, linewidth=1, linestyle="-.", color=TRAINING_COLOR[1], label="Without pretraining (2nd Best)")

    plt.legend(fontsize=10)
    plt.title(f"The {category} energy for each step")
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)
    plt.savefig(f"../output/pretraining-{category}.pdf")


if __name__ == '__main__':
    plot("min")
    plt.clf()
    plot("mean")
