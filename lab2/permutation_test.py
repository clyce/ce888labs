import numpy as np


def permutation_test_(dataset, target_t):
    permuted = np.random.permutation(
        np.concatenate(dataset))
    pold = permuted[:len(dataset[0])]
    pnew = permuted[len(dataset[0]):]
    return (np.mean(pnew) - np.mean(pold)) > target_t


def permutation_test(iterations, dataset, target_t):
    total = [permutation_test_(dataset, target_t) for _ in range(iterations)]
    extreme = [b for b in total if b]
    return len(extreme) / len(total)
