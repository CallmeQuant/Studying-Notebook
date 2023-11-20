import numpy as np
import pandas as pd
from save_load import load_model
def backtest_models(models, names, checkpoint_path, test_data, test_target, times_te, config, device):
    # Initialize dictionaries to store the weights and portfolios for each model
    model_weights = {}
    model_portfolios = {}

    # Initialize a list to store the equal weight portfolio
    equal_portfolio = [10000]
    equal_weights = np.ones(config['N_FEAT']) / config['N_FEAT']

    for model, model_name in zip(models, names):
        # Load the model
        model = load_model(model, checkpoint_path, model_name)

        # Initialize a list to store the weights for model
        weights = []
        # Initialize a list to store the portfolio for model
        portfolio = [10000]

        # Test data should be torch.Tensor
        for i in range(0, test_data.shape[0], config["LEN_PRED"]):
            x = test_data[i][np.newaxis, :, :]
            out = model(x.float().to(device))[0]
            weights.append(out.detach().cpu().numpy())
            m_rtn = np.sum(test_target[i], axis=0)
            portfolio.append(
                portfolio[-1] * np.exp(np.dot(out.detach().cpu().numpy(), m_rtn))
            )
            if model_name == names[0]:  # Only calculate the equal weight portfolio for the first model
                equal_portfolio.append(
                    equal_portfolio[-1] * np.exp(np.dot(equal_weights, m_rtn))
                )


        # Store the weights and portfolio for this model
        model_weights[model_name] = weights
        model_portfolios[model_name] = portfolio

    # Get the time indices
    idx = np.arange(0, len(times_te), 11)
    model_portfolios['date'] = pd.to_datetime(times_te, unit = 'ns')[idx]
    model_portfolios['EWP'] = equal_portfolio

    return model_weights, model_portfolios

def compute_mdd(portfolio_values):
    # Calculate the running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    # Ensure the value never drops below 1
    running_max[running_max < 10000] = 10000
    # Calculate the drawdown
    drawdown = (portfolio_values / running_max) - 1
    # Return the minimum (i.e., the maximum drawdown)
    return drawdown.min()

def compute_metrics(performance):
    # Initialize a dictionary to store the results
    results = {}

    # Loop over each column in the performance DataFrame
    for column in performance.columns:
        # Copy the performance DataFrame
        result = performance.copy()

        # Calculate the return for this column
        result[column + "_Return"] = np.log(result[column]) - np.log(result[column].shift(1))
        result = result.dropna()

        # Calculate the expected return
        expected_return = result[column + "_Return"].mean() * 12

        # Calculate the volatility
        volatility = result[column + "_Return"].std() * np.sqrt(12)

        # Calculate the Sharpe ratio
        sharpe_ratio = expected_return / volatility

        # Calculate the maximum drawdown
        mdd = compute_mdd(result[column])

        # Store the results
        results[column] = {
            'Annualized Return': expected_return,
            'Annualized Volatility': volatility,
            'Annualized Sharpe Ratio': sharpe_ratio,
            'MDD': mdd
        }

    # Convert the results dictionary to a DataFrame
    results_df = pd.DataFrame(results).T

    return results_df

def create_dataset(data, data_len, n_stock):
    times = []
    dataset = np.array(data.iloc[:data_len, :]).reshape(1, -1, n_stock)
    times.append(data.iloc[:data_len, :].index)

    for i in range(1, len(data) - data_len + 1):
        addition = np.array(data.iloc[i : data_len + i, :]).reshape(1, -1, n_stock)
        dataset = np.concatenate((dataset, addition))
        times.append(data.iloc[i : data_len + i, :].index)
    return dataset, times


def data_split(data, train_len, pred_len, tr_ratio, n_stock):
    return_train, times_train = create_dataset(
        data[: int(len(data) * tr_ratio)], train_len + pred_len, n_stock
    )
    return_test, times_test = create_dataset(
        data[int(len(data) * tr_ratio) :], train_len + pred_len, n_stock
    )

    x_tr = np.array([x[:train_len] for x in return_train])
    y_tr = np.array([x[-pred_len:] for x in return_train])
    times_tr = np.unique(
        np.array([x[-pred_len:] for x in times_train]).flatten()
    ).tolist()

    x_te = np.array([x[:train_len] for x in return_test])
    y_te = np.array([x[-pred_len:] for x in return_test])
    times_te = np.unique(
        np.array([x[-pred_len:] for x in times_test]).flatten()
    ).tolist()

    return x_tr, y_tr, x_te, y_te, times_tr, times_te

