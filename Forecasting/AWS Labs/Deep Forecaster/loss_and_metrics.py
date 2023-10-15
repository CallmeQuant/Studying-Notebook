import numpy as np
import pandas as pd
import torch

class Loss():
    def __init__(self, ytrue):
        self.ytrue = ytrue

    def gaussian_likelihood(self, mu, sigma):
        nll = torch.log(sigma + 1) + (self.ytrue - mu) ** 2 / (2 * sigma)
        return nll.mean()

    def negative_binomial_likelihood(self, mu, alpha):
        batch_size, seq_len = self.ytrue.size()
        loglik = torch.lgamma(self.ytrue + 1. / alpha) - torch.lgamma(self.ytrue + 1) - torch.lgamma(1. / alpha) - \
                 1. / alpha * torch.log(1 + alpha * mu) + self.ytrue * torch.log(alpha * mu / (1 + alpha * mu))

        return -loglik.mean()



def MAPE(y,f, is_symmetric = False):
  """

  :param y: array of true values
  :param f: array of predicted values
  :param is_symmetric: If true, return the Symmetric version of MAPE (see i.e.,
  :return: Mean absolute percentage error
  """
  y = np.ravel(y) + 1e-4
  f = np.ravel(f)
  if is_symmetric:
      return 100 * np.mean(2 * np.abs((y-f) / (y+f)))
  else:
      return 100 * np.mean(np.abs((y-f)) / y)

def MAE(y, f):
    """
    :param y: array of true values
    :param f: array of predicted values
    :return: Mean Absolute Error
    """
    y = np.ravel(y)
    f = np.ravel(f)
    return np.mean(np.abs(y-f))

def MSE(y, f):
    """

    :param y: array of true values
    :param f: array of predicted values
    :return: Mean Squared Error
    """
    y = np.ravel(y)
    f = np.ravel(f)

    return np.mean((y-f)**2)

def RMSSE(y, f):
    """

    :param y: array of true values
    :param f: array of predicted values
    :return: Root Mean Scaled Squared Error
    """
    y = np.ravel(y)
    f = np.ravel(f)
    denom = np.mean((y[1:] - y[:-1]) ** 2)
    numer = MSE(y, f)

    return np.sqrt(numer / denom)

def MASE(y, f):
    """

    :param y: array of true values
    :param f: array of predicted values
    :return: Root Mean Scaled Squared Error
    """
    y = np.ravel(y)
    f = np.ravel(f)
    nom = np.sum(np.abs(f - y))
    denom = y.shape[0] / (y.shape[0] - 1) * np.sum(np.abs( y[1:] - y[:-1] ))
    loss = nom / denom
    return loss


def MQloss(y, f_samps, model = None, qs = None):
  # Compute the quantiles of the forecasts
  quantiles = []
  if qs is None:
    qs = np.arange(0, 1.01, 0.01)
  ar = (qs * 100).astype(np.int64)
  for i in ar:
    alpha = (100-i)
    # For MQRNN model, just access the dimension 2 to quantiles
    if model is not None:
      q = f_samps[:, :, i].ravel()
    else:
      q = np.percentile(f_samps, [100 - alpha], axis=1).ravel()
    quantiles.append(q)

  quantiles = np.asarray(quantiles).reshape(-1, len(ar))

  # Compute the multi-quantile loss
  # absolute error: dim (num_periods, ) \hat{y}_{t1}^{q00} - y_{t1} .... \hat{y}_{t1}^{q100} - y_{t1}
  #                                                 ...
  #                                     \hat{y}_{tN}^{q00} - y_{tN} .... \hat{y}_{tN}^{q100} - y_{tN}

  L = np.zeros_like(y.ravel()).astype(np.float64)

  y_ravel = np.swapaxes(y, 1, 0).ravel()
  for i, q in enumerate(qs):
    err = y_ravel - quantiles[:, i]
    L += np.maximum(err, np.zeros_like(err)) * (1 - q) + q * np.maximum(-err, np.zeros_like(err))

  mqloss = (1/y.shape[1]) * L.mean(axis = 0)
  return mqloss

metrics_dict = {
        'mse':MSE,
        'mase': MASE,
        'rmsse': RMSSE,
        'mape':MAPE,
        'mae':MAE,
        'mqloss': MQloss
    }

def evaluate(y, f, f_samps, model = None, metrics = ('mse','mase', 'rmsse', 'mape','mae'), return_dataframe = True):
  results = {}
  for metric in metrics:
      try:
        if metric != 'mqloss':
          y = np.ravel(y)
          f = np.ravel(f)
          results[metric] = metrics_dict[metric](y, f)
        else:
          y = np.expand_dims(y, axis = 0)
          results[metric] = metrics_dict[metric](y, f_samps, model)
      except Exception as err:
          results[metric] = np.nan
          print('Unable to compute metrics {0}: {1}'.format(metric, err))
  if return_dataframe:
    results_df = pd.DataFrame.from_dict(results.items())
    results_df.columns = ['Metric','Value']
    return results_df

  return results