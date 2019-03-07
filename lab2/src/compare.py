import bootstrap
import permutation_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help="path for the data, should be a csv of two columns: `current, new`, each column indicates the score to compare")
    parser.add_argument("--ci", "--confidence-interval", type=int,
                        help="The confidence interval of the bootstrap analysis, default is 95")
    parser.add_argument("--nbs", "--sample-times-bootstrap", type=int,
                        help="The number of times that the bootstrap process samples, defualt is 100000")
    parser.add_argument("--npt", "--sample-times-permutation", type=int,
                        help="The number of times that the permutation test process samples, defualt is 1000")
    parser.add_argument("--acc", "--accept-rate", type=float,
                        help="accept rate of the permutation test, default is 0.05")
    args = parser.parse_args()
    ci = args.ci or 95
    nbs = args.nbs or 100000
    npt = args.npt or 1000
    acc = args.acc or 0.05

    df = pd.read_csv(args.data)

    # (let (... ((juxt first second) ...)) ...)
    fleets = np.array([
        [x
         for x
         in col
         if not math.isnan(x)]
        for col
        in df.values.T])

    ((curr_mean, curr_lower, curr_upper),
     (new_mean,  new_lower,  new_upper)) = [
         bootstrap.bootstrap(ci)(
             data, len(data), nbs)
         for data
         in fleets]

    print("Bootstrap result on ci: ", ci, " after ", nbs, "iterations")
    print("\tcurrent | ",
          "mean: ", curr_mean, "lower: ", curr_lower, "upper: ", curr_upper)
    print("\tnew | ",
          "mean: ", new_mean, "lower: ", new_lower, "upper: ", new_upper)

    target_t = new_mean - curr_mean
    ptest_res_p = permutation_test.permutation_test(npt)(fleets, target_t)
    print(
        "newer is better" if ptest_res_p < acc else "newer makes no sense",
        "since the p is", ptest_res_p)
