import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores_t = np.array(scores)
    mx = np.max(scores_t)
    sm = np.sum(np.exp(scores_t-mx))
    return np.exp(scores_t-mx)/sm