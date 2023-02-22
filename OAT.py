import torch
import torch.optim as optim

from models import Linear
from synthetic_data import GenerateDataset, SynDataset, DatasetFromList
from loss import hinge_loss01, OAT_loss

from tqdm import tqdm
from plot_utils import PlotEval, Eval
import os
import argparse




def train(model, train_loader, loss, optimizer, num_epochs):
    for _ in range(num_epochs):
        for _, (X,y) in enumerate(train_loader):
            X,y = X.to(device), y.to(device)
            pre = model(X).flatten()
            cost = loss(pre, y.flatten())

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()


def AT_SVM(name, model, num_epochs, train_data, eps, batch_size1, batch_size2 = 100, num_epochs2 = 5, lr = 0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 0.01)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size1,
                                            shuffle=False)
    for epoch in tqdm(range(num_epochs)):
        for _, (X,y) in enumerate(train_loader):
            # Generate adversarial example
            N = X.shape[0]
            d = X.shape[1]
            weight = model.state_dict()['linear.0.weight'] 
            sign = 1*(weight > 0) - 1*(weight < 0)
            delta = eps* sign.repeat(N,1) * (y.repeat(d, 1).transpose(1,0)*2 - 1)
            X_adv = (X - delta).detach()

            # Train model
            dataset_i = DatasetFromList([X_adv],[y])
            train_loader_i = torch.utils.data.DataLoader(dataset=dataset_i,
                                                    batch_size=batch_size2,
                                                    shuffle=True)
            
            train(model, train_loader_i, hinge_loss01, optimizer, num_epochs = num_epochs2)

        torch.save(model.state_dict(), 'checkpoint/'+ name + '_epoch_'+ str(epoch))


    
def oat_train(model, train_loader, optimizer, num_epochs, eps):
    for _ in range(num_epochs):
        for _, (X,y) in enumerate(train_loader):
            X,y = X.to(device), y.to(device)
            cost = OAT_loss(model, X, y.flatten(), eps)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

def OAT_SVM(name, model, num_epochs, train_data, eps, batch_size1, batch_size2 = 100, num_epochs2 = 5, lr = 0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 0.01)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=batch_size2,
                                            shuffle=False)
    for epoch in tqdm(range(num_epochs)):
        oat_train(model, train_loader, optimizer, num_epochs= num_epochs2, eps = eps)
        torch.save(model.state_dict(), 'checkpoint/'+ name + '_epoch_'+ str(epoch))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--d', default=2000, type=int, help="Number of non-robust features")
    parser.add_argument('--p', default=0.7, type=float, help="Probability that a robust feature is correct")
    parser.add_argument('--mu', default=1e-2, type=float, help="Mean of non-robust features")
    parser.add_argument('--sigma', default=1e-2, type=float, help="Std of non-robust features")
    parser.add_argument('--eps', default=2e-2, type=float, help="Perturbation budget") 
    args = parser.parse_args()

    train_size = 10000
    test_size = 10000

    GenerateDataset(args.p,args.mu,args.sigma, args.d, train_size, test_size)
    syn_train = SynDataset(train = True)
    syn_test = SynDataset(train = False)

    batch_size1 = train_size
    batch_size2 = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader = torch.utils.data.DataLoader(dataset=syn_test,
                                            batch_size=batch_size1,
                                            shuffle=False)
    try:
        print('--- Creating a directory ---')
        os.mkdir('checkpoint')
    except:
        print('--- The directory exists ---')

    # Adversarial training
    model = Linear(in_feature= args.d+1, out_feature= 1)
    AT_SVM('AT_SVM', model, 50, syn_train, eps = args.eps, batch_size1 = train_size, 
    batch_size2 = 200, num_epochs2 = 5, lr = 0.01)

    

    # Optimal adversarial training
    model = Linear(in_feature= args.d+1, out_feature= 1)
    OAT_SVM('OAT_SVM', model, 50, syn_train, eps = args.eps, batch_size1 = train_size, 
    batch_size2 = 200, num_epochs2 = 1, lr = 0.01)


    # Plot
    PlotEval('AT_SVM', 50, Linear(in_feature= args.d+1, out_feature = 1), test_loader, eps = args.eps)
    PlotEval('OAT_SVM', 50, Linear(in_feature= args.d+1, out_feature = 1), test_loader, eps = args.eps)

    # Eval
    # result = {'d': args.d, 'AT_SVM': Eval('AT_SVM', 50, Linear(in_feature= args.d+1, out_feature = 1), test_loader, eps = args.eps),
    # 'OAT_SVM': PlotEval('OAT_SVM', 50, Linear(in_feature= args.d+1, out_feature = 1), test_loader, eps = args.eps)}
    