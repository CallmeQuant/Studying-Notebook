import torch
import numpy as np
def max_sharpe(y_return, weights, regularization = False, lambda_=0.001, C=1.002):
    weights = torch.unsqueeze(weights, 1)
    meanReturn = torch.unsqueeze(torch.mean(y_return, axis=1), 2)
    if torch.cuda.is_available():
      covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return]).to('cuda')
    else:
      covmat = torch.Tensor([np.cov(batch.cpu().T, ddof=0) for batch in y_return])
    portReturn = torch.matmul(weights, meanReturn)
    portVol = torch.matmul(
        weights, torch.matmul(covmat, torch.transpose(weights, 2, 1))
    )
    # Add a small constant to the denominator to avoid division by zero
    portVol = torch.where(portVol < 1e-6, torch.ones_like(portVol) * 1e-6, portVol)
    objective = (portReturn * 12 - 0.02) / (torch.sqrt(portVol * 12)) # risk-free rate = 0.02
    # Add a penalty term to the objective function to restrict too many stocks
    # being included in one portfolio
    penalty = lambda_ * (torch.abs(C - weights.sum(dim=1)) * weights).mean()
    if regularization:
      return -(objective.mean() - penalty)
    else:
      return -objective.mean()

def equal_risk_parity(y_return, weights):
    B = y_return.shape[0]
    F = y_return.shape[2]
    weights = torch.unsqueeze(weights, 1).to("cuda")
    covmat = torch.Tensor(
        [np.cov(batch.cpu().T, ddof=0) for batch in y_return]
    )  # (batch, 50, 50)
    covmat = covmat.to("cuda")
    sigma = torch.sqrt(
        torch.matmul(weights, torch.matmul(covmat, torch.transpose(weights, 2, 1)))
    )
    mrc = (1 / sigma) * (covmat @ torch.transpose(weights, 2, 1))
    rc = weights.view(B, F) * mrc.view(B, F)
    target = (torch.ones((B, F)) * (1 / F)).to("cuda")
    risk_diffs = rc - target
    sum_risk_diffs_squared = torch.mean(torch.square(risk_diffs))
    return sum_risk_diffs_squared