import torch
import torch.nn as nn
from torch.utils.data import Dataset

def GenerateData(p,mu, sigma,d,N):

    def GenerateGaussian(p, mu, sigma, d, N):
        feature_1 = torch.bernoulli(torch.ones((N,1))*p)*2 - 1
        feature_j = torch.normal(mu, sigma, size=(N, d))
        return torch.hstack((feature_1, feature_j))

    
    x_1 = GenerateGaussian(p,mu,sigma,d,N)
    x_0 = -GenerateGaussian(p,mu,sigma,d,N)
    
    xy_0 = torch.hstack((torch.zeros((N,1)),x_0))
    xy_1 = torch.hstack((torch.ones((N,1)),x_1))
    xy = torch.vstack((xy_0,xy_1))
    return xy

def GenerateDataset(p,mu,sigma, d,train_size, test_size):
    train_data = GenerateData(p,mu,sigma, d,train_size)
    test_data = GenerateData(p,mu,sigma, d,test_size)
    torch.save(train_data, './data/synthetic/train_data.pth')
    torch.save(test_data, './data/synthetic/test_data.pth')

class SynDataset(Dataset):
    def __init__(self, train):
        if train:
            xy = torch.load('./data/synthetic/train_data.pth')
        else:
            xy = torch.load('./data/synthetic/test_data.pth')
        self.x = xy[:,1:]
        self.y = xy[:, 0].type(torch.long)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

class DatasetFromList(Dataset):
    def __init__(self, list_x, list_y):
        self.x = torch.vstack(list_x)
        self.y = torch.hstack(list_y)
        self.n_samples = torch.vstack(list_x).shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples 

class DatasetFromListMatch(Dataset):
    def __init__(self, list_x, list_y):
        self.x = torch.stack(list_x)
        self.y = torch.stack(list_y)
        self.n_samples = list_y[0].shape[0]

    def __getitem__(self, index):
        return self.x[:,index,:], self.y[:,index]

    def __len__(self):
        return self.n_samples
