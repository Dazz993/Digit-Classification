import torch


def logistic_regression_loss(input, target):
    input = input.reshape(-1)
    target = target.reshape(-1)
    assert len(input) == len(target)

    return - (target * input - torch.log(1 + torch.exp(input))).mean()


def regularization_loss(model, lam=0.01, p=2):
    assert p == 1 or p == 2
    weight_list = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_list.append(param)

    loss = 0
    for weight in weight_list:
        loss += _compute_ridge(weight) if p == 2 else _compute_lasso(weight)

    return lam * loss

def _compute_lasso(tensor):
    return torch.sum(torch.abs(tensor))

def _compute_ridge(tensor):
    return torch.sum(tensor ** 2)