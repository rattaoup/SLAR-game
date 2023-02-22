import torch

def hinge_loss(y_prime,y):
    return (torch.max(torch.zeros_like(y_prime), 1 - y_prime*y)).type(torch.float).mean()

def hinge_loss01(y_prime,y):
    return hinge_loss(y_prime, 2*y-1)

def OAT_loss(model, x, y, eps):
    y_prime = model(x).flatten()
    l1_norm = torch.norm([param for param in model.linear.parameters()][0], p = 1)
    return (torch.max(torch.zeros_like(y_prime), 1 - y_prime*(2*y-1) + eps*l1_norm)).type(torch.float).mean()
