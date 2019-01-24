import bootstrap
import permutation_test

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

if __name__ == "__main__":
    df = pd.read_csv('./vehicles.csv')

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
         bootstrap.bootstrap(95)(
             data, len(data), 100000)
         for data
         in fleets]

    print("current | ",
          "mean: ", curr_mean, "lower: ", curr_lower, "upper: ", curr_upper)
    print("new | ",
          "mean: ", new_mean, "lower: ", new_lower, "upper: ", new_upper)

    target_t = new_mean - curr_mean
    ptest_res_p = permutation_test.permutation_test(
        1000000,
        fleets,
        target_t)
    print(
        "newer is better" if ptest_res_p < 0.05 else "newer makes no sense",
        "since the p is", ptest_res_p)
