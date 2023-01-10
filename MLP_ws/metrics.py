import numpy as np

def APk(predict, target, k=50):
    predict = predict[:k]
    score = 0.0
    num_hits = 0.0
    for idx, p in enumerate(predict):
        if p in target and p not in predict[:idx]:
            num_hits += 1.0
            score += num_hits / (idx + 1.0)
    if not target:
        return 0.0
    return score / min(len(target), k)

def mAPk(predict, target, k=50):
    """
    predict:[instance num, predicted class number for each instance]
        a list of lists of prediction classes
    target:[instance num, target class number for each instance]
        a list of lists of target classes
    """
    return np.mean([APk(p, t, k) for p, t in zip(predict, target)])
