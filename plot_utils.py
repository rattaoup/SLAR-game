import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Eval(name, num_epochs, model_cls, test_loader, eps):
    # Accuracy
    robust_acc = []
    standard_acc = []
    for epoch in range(num_epochs):
        model = model_cls
        model.load_state_dict(torch.load('checkpoint/'+ name + '_epoch_'+ str(epoch)))
        # SVM loss
        standard_acc.append(StandardAccSVM(model, test_loader))
        robust_acc.append(CertifiedRobustAcc(model, test_loader, eps))


    # Convergence
    weight = []
    for epoch in range(num_epochs):
        weight.append(torch.load('checkpoint/'+ name + '_epoch_'+ str(epoch)))
    weight_diff = ([float((weight[i+1]['linear.0.weight'] - weight[i]['linear.0.weight']).norm(p = 2)) for i in range(len(weight)-1)])

    return {'Standard acc':standard_acc, 'Robust acc':robust_acc, 'Weight diff': weight_diff }


def PlotEval(name, num_epochs, model_cls, test_loader, eps):
    # Accuracy
    robust_acc = []
    standard_acc = []
    for epoch in range(num_epochs):
        model = model_cls
        model.load_state_dict(torch.load('checkpoint/'+ name + '_epoch_'+ str(epoch)))
        # SVM loss
        standard_acc.append(StandardAccSVM(model, test_loader))
        robust_acc.append(CertifiedRobustAcc(model, test_loader, eps))


    # Convergence
    weight = []
    for epoch in range(num_epochs):
        weight.append(torch.load('checkpoint/'+ name + '_epoch_'+ str(epoch)))
    weight_diff = ([(weight[i+1]['linear.0.weight'] - weight[i]['linear.0.weight']).norm(p = 2) for i in range(len(weight)-1)])

    # Plot
    f, axs = plt.subplots(1,2,figsize=(8,3), dpi = 100)
    axs[0].plot(standard_acc)
    axs[0].plot(robust_acc)
    axs[0].set_xlabel('Epochs')
    axs[0].set_title('Accuracy of '+ name)
    axs[0].legend(['Standard Acc', 'Certified Robust Acc'])
    
    axs[1].plot(weight_diff)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('||w_{t+1} - w_t||_2')
    axs[1].set_title('Convergence of '+ name)
    plt.tight_layout()
    plt.show()


def StandardAccSVM(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    for X, y in test_loader:
        
        X = X.to(device)
        outputs = model(X)
        total += y.size(0)
        correct += ((outputs.reshape(-1,1) > 0).reshape(-1) == y).type(torch.float).sum()
        
    return (100 * float(correct) / total)

def CertifiedRobustAcc(model, test_loader, eps):
    model.eval()

    robust = 0
    total = 0

    for X, y in test_loader:
        dist = torch.abs((model(X)/model.state_dict()['linear.0.weight'].norm(p = 1)))
        outputs = model(X)
        total += y.size(0)
        robust += (((dist > eps).type(torch.float)).reshape(-1)*((outputs.reshape(-1,1) > 0).reshape(-1) == y).type(torch.float).reshape(-1)).sum()
        
    return (100 * float(robust) / total)