import math

class descriptive():
  def __init__(self, x):
    self.obs = x

  def mean(self):
    return sum(self.obs) / len(self.obs)

  def median(self):
    sorted_obs = sorted(self.obs)
    n = len(sorted_obs)

    if n % 2 == 1:
      return sorted_obs[n // 2]
    else:
      return (sorted_obs[n // 2] + sorted_obs[n // 2 - 1]) / 2

  def mode(self):
    counts = {}
    for x in self.obs:
      if x in counts:
        counts[x] += 1
      else:
        counts[x] = 1

    max_count = max(counts.values())
    modes = [x for x, count in counts.items() if count == max_count]

    return modes[0]

  def sample_variance(self):
    mean = self.mean()
    sample_var = sum([(x - mean)**2 for x in self.obs]) / (len(self.obs)  -1 )

    return sample_var

  def sample_standard_deviation(self):
    sample_var = self.sample_variance()
    return math.sqrt(sample_var)

# Testing 
from statistics import stdev, variance, mean, mode, median
import random

random.seed(28)
salaries = [round(random.random()*1000000, -3) for _ in range(100)]
descriptive_stats = descriptive(salaries)

descriptive_stats.mean() == mean(salaries)
descriptive_stats.median() == median(salaries)
descriptive_stats.mode() == mode(salaries)
descriptive_stats.sample_variance() == variance(salaries)
descriptive_stats.sample_standard_deviation() == stdev(salaries)
