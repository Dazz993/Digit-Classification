import torch


def logistic_regression_loss(input, target):
    input = input.reshape(-1)
    target = target.reshape(-1)
    assert len(input) == len(target)

    return - (target * input - torch.log(1 + torch.exp(input))).mean()


def regularization_loss(model, lam=0.01, p=2):
    weight_list = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_list.append(param)

    weights = torch.tensor(weight_list)
    loss = torch.norm(weights, p=p)

    return lam * loss