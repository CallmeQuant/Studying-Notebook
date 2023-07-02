def update_normal(prior_mean, prior_var, data, known_var):
  """
  Estimating normal distribution parameters with unknown `mean` and known `variance`

  Parameters
  ----------
  prior_mean:  Mean of the prior distribution
  prior_var:  Variance of the prior distribution
  data: Obsevered data. Expect to be array-like 
  known_var: Variance of the data generating distribution (assumed known)

  Returns
  -------
  Mean and variance of the posterior distribution
  """
  n = len(data)
  if not isinstance(data, np.ndarray):
    raise ValueError("""Data must be in {} format. {} is provided""".format(
        np.ndarray, type(data))
    )
    data = np.array(data)
  
  data_mean = np.mean(data)

  post_var = 1 / (n / known_var + 1 / prior_var)
  post_mean = post_var * (n * data_mean / known_var + prior_mean / prior_var)

  return post_mean, post_var
