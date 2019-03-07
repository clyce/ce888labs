import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import argparse

# matplotlib.use('Agg')


def bootstrap(ci):

    def bootstrap_(sample, sample_size, iterations):
        subsamples = np.random.choice(sample, size=(sample_size, iterations))
        data_mean = np.mean(subsamples)
        iter_means = [np.mean(i) for i in subsamples]
        (lower, upper) = [
            np.percentile(iter_means, ci_)
            for ci_
            in ((100 - ci), ci)]
        return data_mean, lower, upper

    return bootstrap_


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="path for the data")
    parser.add_argument("--ci", "--confidence-interval", type=int,
                        help="The confidence interval of the bootstrap analysis, default is 95")
    parser.add_argument("--out", "--output-file", type=str,
                        help="output filename, default to `out/bootstrap_confidence.png`")
    args = parser.parse_args()
    ci = args.ci or 95
    output = args.out or 'out/boostrap_confidence.png'

    df = pd.read_csv(args.data)

    data = df.values.T[1]
    boots = []
    for i in range(100, 100000, 1000):
        boot = bootstrap(95)(data, data.shape[0], i)
        boots.append([i, boot[0], "mean"])
        boots.append([i, boot[1], "lower"])
        boots.append([i, boot[2], "upper"])

    df_boot = pd.DataFrame(
        boots, columns=['Boostrap Iterations', 'Mean', "Value"])
    sns_plot = sns.lmplot(
        df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

    means = df_boot.iloc[:, 1]
    print(min(means), max(means))
    sns_plot.axes[0, 0].set_ylim(min(means), max(means))
    sns_plot.axes[0, 0].set_xlim(0, 100000)

    sns_plot.savefig(output, bbox_inches='tight')

    # print ("Mean: %f")%(np.mean(data))
    # print ("Var: %f")%(np.var(data))
