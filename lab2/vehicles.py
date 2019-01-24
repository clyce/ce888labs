import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')

    sns_plot = sns.lmplot(df.columns[0], df.columns[1], data=df, fit_reg=False)

    sns_plot.axes[0, 0].set_ylim(0,)
    sns_plot.axes[0, 0].set_xlim(0,)

    sns_plot.savefig("vehicle_scatterplot.png", bbox_inches='tight')

    data_curr = df.values.T[0]
    data_new = [x
                for x
                in df.values.T[1]
                if not math.isnan(x)]

    plt.clf()

    sns_plot = sns.distplot(data_curr).get_figure()
    sns_plot = sns.distplot(data_new).get_figure()

    axes = plt.gca()
    axes.set_xlabel('MPG')
    axes.set_ylabel('Count')

    sns_plot.savefig("vehicle_histograms.png", bbox_inches='tight')
