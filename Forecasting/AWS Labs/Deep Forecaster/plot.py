import matplotlib.pyplot as plt
import numpy as np


def plot_data_forecast(params, y_true_plot, f_samps=None, model=None, conditional_forecast='mean', *args):
  fig, ax = plt.subplots(1, 1, figsize=(20, 5))
  x_values = [k + params['sequence_length'] + params['prior'] - params['sequence_length'] \
              for k in range(params['sequence_length'])]

  # Check conditions to plot forecasts of other models
  if model != 'MQRNN':
    if f_samps is None:
      raise ValueError('Need to pass `f_samps` for other forecasting models')

    if conditional_forecast == 'mean':
      f = np.mean(f_samps, axis=1)
    elif conditional_forecast == 'median':
      f = np.quantile(f_samps, 0.5, axis=1)
    else:
      raise NotImplementedError(
        f'Should be `mean` or `median` forecast, given unknown forecast type {conditional_forecast}')
    # Plot median forecast
    ax.plot(x_values, f, color="k", label=(str(conditional_forecast)).capitalize() + " forecast")

    # Plot credible intervals
    if params['probabilistic_CI']:
      ar = np.arange(0, 101, 1)
      ci_legend = "Full range of credible intervals"
    else:
      # Default 95% credible interval
      ar = [95]
      ci_legend = str(params['credible_interval']) + "% credible interval"

    for i in ar:
      alpha = (100 - i) / 2
      upper = np.percentile(f_samps, [100 - alpha], axis=1)
      lower = np.percentile(f_samps, [alpha], axis=1)
      fill = ax.fill_between(x=x_values, y1=lower.ravel(), y2=upper.ravel(), alpha=0.3, color='blue')

  # Since MQRNN not produce full probabilistic samples, we have to pass its own forecasts to plot
  else:
    y_pred = args[0] if args else None
    if y_pred is None:
      raise ValueError('Need to pass forecasts of MQRNN to plot')

    # Plot median forecast
    ax.plot(x_values, y_pred[:, :, int(np.floor(params['num_quantiles'] / 2))].ravel(), color="k",
            label="Median forecast")

    # Plot credible intervals
    ci_legend = str(params['credible_interval']) + "% credible interval"

    fill = ax.fill_between(x=x_values, y1=y_pred[:, :, 0].ravel(),
                           y2=y_pred[:, :, params['num_quantiles'] - 1].ravel(),
                           alpha=0.3, color='blue')

    # Add legend for credible intervals only once
    fill.set_label(ci_legend)

  # Plot observations
  yplot = y_true_plot[-1, -params['sequence_length'] - params['prior']:]
  ax.scatter(range(len(yplot)), yplot, color='k', label="Observations")

  # Plot forecast start date
  ymin, ymax = plt.ylim()
  ax.vlines(params['sequence_length'] + params['prior'] - params['sequence_length'],
            ymin, ymax, color="green", linestyles="dashed",
            linewidth=2, label="Forecast start date")

  ax.set_ylim(ymin, ymax)
  ax.set_xlabel("Time")
  ax.set_ylabel("Y")

  # Create legend
  ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=True)

  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

  plt.tight_layout()
  plt.show()