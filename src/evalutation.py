from numba import jit
import numpy as np
import warnings
warnings.filterwarnings("ignore")

@jit
def qwk(true, pred, max_rat=3):
    # Quadratic Weighted Kappa (QWK)
    assert(len(true) == len(pred))
    true = np.asarray(true, dtype=int)
    pred = np.asarray(pred, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(true.shape[0]):
        i, j = true[k], pred[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / true.shape[0]

    return 1 - o / e
