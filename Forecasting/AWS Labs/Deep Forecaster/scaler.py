import numpy as np

class Normalizer:
    """
    Wrapper object to perform normalization
    """
    def __init__(self):
        self.mu = None
        self.std = None

    def fit_transform(self, x):
        self.mu = np.mean(x)
        self.std = np.std(x) + 1e-4 # ddof = 1 implies sample std where ddof stands for degree of freedom
        x_normalized = (x - self.mu) / self.std

        return x_normalized

    def inverse_transform(self, x):
        return (x * self.std) + self.mu

    def transform(self, x):
        return (x - self.mu) / self.std


class MeanCenter:
    """
    Wrapper object to perform mean neutralization
    """
    def __init__(self):
        self.mu = None

    def fit_transform(self, x):
        self.mu = np.mean(x)
        return x / self.mu

    def inverse_transform(self, x):
        return x * self.mu

    def transform(self, x):
        return x / self.mu


class LogScaler:
    """
    Wrapper object to perform logarithmic scaling
    """
    def fit_transform(self, x):
        return np.log1p(x)

    def inverse_transform(self, x):
        return np.expm1(x)

    def transform(self, x):
        return np.log1p(x)

class MaxScaler:
    """
    Wrapper object to perform max neutralization
    """
    def __init__(self):
        self.max = None

    def fit_transform(self, x):
        self.max = np.max(x)
        return x / self.max

    def inverse_transform(self, x):
        return x * self.max
    def transform(self, x):
        return x / self.max