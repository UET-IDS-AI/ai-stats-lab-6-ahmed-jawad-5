import math
import numpy as np


def _validate_bernoulli_data(data):
    if data is None or len(data) == 0:
        raise ValueError("Data is empty")

    arr = np.array(data)

    if not np.all(np.isin(arr, [0, 1])):
        raise ValueError("Bernoulli data must contain only 0 and 1")

    return arr


def bernoulli_log_likelihood(data, theta):
    if theta <= 0 or theta >= 1:
        raise ValueError("Theta must be in (0,1)")

    arr = _validate_bernoulli_data(data)

    successes = np.sum(arr)
    failures = len(arr) - successes

    return successes * np.log(theta) + failures * np.log(1 - theta)


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    arr = _validate_bernoulli_data(data)

    n = len(arr)
    successes = int(np.sum(arr))
    failures = n - successes

    mle = successes / n

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    log_likelihoods = {}

    for theta in candidate_thetas:
        ll = bernoulli_log_likelihood(arr, theta)
        log_likelihoods[theta] = ll

    best_candidate = None
    best_value = -np.inf

    for theta in candidate_thetas:
        if log_likelihoods[theta] > best_value:
            best_value = log_likelihoods[theta]
            best_candidate = theta

    return {
        "mle": float(mle),
        "num_successes": successes,
        "num_failures": failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }


def _validate_poisson_data(data):
    if data is None or len(data) == 0:
        raise ValueError("Data is empty")

    arr = np.array(data)

    if np.any(arr < 0):
        raise ValueError("Counts must be non-negative")

    if not np.all(np.floor(arr) == arr):
        raise ValueError("Counts must be integers")

    return arr.astype(int)


def poisson_log_likelihood(data, lam):
    if lam <= 0:
        raise ValueError("Lambda must be > 0")

    arr = _validate_poisson_data(data)

    ll = 0.0
    for x in arr:
        ll += x * np.log(lam) - lam - math.lgamma(x + 1)

    return ll


def poisson_mle_analysis(data, candidate_lambdas=None):
    arr = _validate_poisson_data(data)

    n = len(arr)
    total = int(np.sum(arr))
    sample_mean = total / n

    mle = sample_mean

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    log_likelihoods = {}

    for lam in candidate_lambdas:
        ll = poisson_log_likelihood(arr, lam)
        log_likelihoods[lam] = ll

    best_candidate = None
    best_value = -np.inf

    for lam in candidate_lambdas:
        if log_likelihoods[lam] > best_value:
            best_value = log_likelihoods[lam]
            best_candidate = lam

    return {
        "mle": float(mle),
        "sample_mean": float(sample_mean),
        "total_count": total,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }
