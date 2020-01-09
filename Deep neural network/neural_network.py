### FDR control for deep neural networks
import torch
import torchvision
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torch.utils import data
import math
from torch.autograd import Variable
import scipy.stats
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore", message = "Numerical issues were encountered ")

dtype = torch.float
### number of true features
nz = 30
### number of features
pset = [500, 1000, 1500, 2000, 3000]
### sample size
n = 1000
### signal strength for the power function and the exponential function
sg = 20*np.sqrt(np.log(p)/n)
### noise level for the power function
ns = 1
### noise level for the exponential function and the sigmoid function
# ns = 0.1
### pairwise correlation between features
rho = 0.5
### FDR control level
q = 0.1
num_split = 50
### algorithmic settings for training dnn
epochs = 500
batch_size = 128
learning_rate = 1e-3

### link function of the single-index model
### power function
def f(t):
  return t**3/2
### exponential function
def f(t):
	return np.exp(t/10)
### sigmoid function
def f(t):
	return 1/(1 + np.exp(-t))


### generate data
def grt(n, p, nz, Sigma):      
    ### sample the true feature index
    nonzero = random.sample(range(0, p), nz)
    zero = np.setdiff1d(range(0, p), nonzero)
    beta = np.zeros(p)
    for nzr in nonzero:
        ### beta for the power function and the exponential function
        beta[nzr] = sg * np.random.randn(1)
        ### beta for the sigmoid function
        # beta[nzr] = np.random.uniform(0.5, 1)
    x_train = np.mat(np.random.multivariate_normal(mean = np.zeros(p), cov = Sigma, size = n))
    x_train = x_train.astype('float32')
    x_train = preprocessing.scale(x_train)
    z_train = np.array(np.dot(x_train, beta)).reshape(n, )

    return z_train, nonzero, zero, x_train

  
### calculate the mirror statistics: the influence function approach
def get_m(model1, model2, x_train1, x_train2):
    w1, w2 = 0, 0

    for i in range(int(n/2)):
        input_ = x_train1[i,:]
        input_.requires_grad = True
        output1 = model1(input_)
        output1.backward()
        w1 += input_.grad
    for i in range(int(n/2)):
        input_ = x_train2[i,:]
        input_.requires_grad = True
        output2 = model2(input_)
        output2.backward()
        w2 += input_.grad

    w1 = w1/(n/2)
    w2 = w2/(n/2)
    w1 = w1.numpy()
    w2 = w2.numpy()

    ### calculate the mirror statistics
    M = np.abs(w1 + w2) - np.abs(w1 - w2)

    return M


### calculate the mirror statistics: the weight multiplication approach
def get_weight(model):
    w1 = list(model.parameters())[0].detach().numpy()
    w2 = list(model.parameters())[2].detach().numpy()
    w3 = list(model.parameters())[4].detach().numpy()

    return np.dot(np.dot(w3, w2), w1)

 
### select the relevant features using mirror statistics
def analys(mm, ww, q):
    ### mm: mirror statistics
    ### ww: absolute value of mirror statistics
    ### q:  FDR control level
    cutoff_set = np.array([max(ww)])
    for t in ww:
        ps = len(mm[mm > t])
        ng = len(mm[mm < -t])
        rto = (float(ng)) / max(ps, 1)
        if rto < q:
            cutoff_set = np.append(cutoff_set, t)
    cutoff = min(cutoff_set)

    selected_index = np.where(mm > cutoff)
    selected_index = list(selected_index)
    selected_index = selected_index[0]

    return selected_index


### calculate fdp and power
def fdp_power(selected_index):
    inc = np.intersect1d(nonzero, selected_index)
    td = len(inc)

    fdp = float(len(selected_index) - td) / max(len(selected_index),1)
    power = float(td) / nz

    return fdp, power


class Dataset(data.Dataset):
    def __init__(self, x, labels):
        'Initialization'
        self.labels = labels
        self.x = x
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)
    def __getitem__(self, index):
        'Generates one sample of data'
        ### select sample
        X = self.x[index]
        y = self.labels[index]

        return X, y


class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fw = nn.Sequential(
                nn.Linear(p, Hidden[0]),
                nn.Sigmoid(),
                nn.Linear(Hidden[0], Hidden[1]),
                nn.Sigmoid(),
                nn.Linear(Hidden[1], 1),
            )
        def forward(self, x):
            return self.fw(x)


def DS(X, z, q):
    inclusion_rate = np.zeros((num_split, p))
    num_select = [0]*num_split

    for it in range(num_split):
        sample_index = np.arange(n)
        np.random.shuffle(sample_index)
        sample1 = sample_index[:int(n/2)]
        sample2 = sample_index[int(n/2):]
        
        model1 = Net()
        optimizer1 = torch.optim.Adam(
        model1.parameters(), lr = learning_rate, weight_decay = 1e-5)
        criterion = nn.MSELoss()
        x_train1 = torch.tensor(X[:int(n/2),:], dtype = dtype)
        z_train1 = torch.tensor(z[:int(n/2)], dtype = dtype)
        dataset1 = Dataset(x_train1, z_train1)
        dataloader1 = DataLoader(dataset1, batch_size = batch_size, shuffle = True)
        model1.train()

        for epoch in range(epochs):
            for data, labels in dataloader1:
                output = torch.squeeze(model1(data))
                l1_regularization = 0
                for i, param in enumerate(model1.parameters()):
                    if i == 0:
                        l1_regularization += torch.norm(param, 1)
                loss = criterion(output, labels) + c*l1_regularization
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()


        model2 = Net()
        optimizer2 = torch.optim.Adam(
        model2.parameters(), lr = learning_rate, weight_decay = 1e-5)
        x_train2 = torch.tensor(X[int(n/2):,:], dtype = dtype)
        z_train2 = torch.tensor(z[int(n/2):], dtype = dtype)
        dataset2 = Dataset(x_train2, z_train2)
        dataloader2 = DataLoader(dataset2, batch_size = batch_size, shuffle = True)
        model2.train()

        for epoch in range(epochs):
            for data, labels in dataloader2:
                output = torch.squeeze(model2(data))
                l1_regularization = 0
                for i, param in enumerate(model2.parameters()):
                    if i == 0:
                        l1_regularization += torch.norm(param, 1)
                loss = criterion(output, labels) + c*l1_regularization
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

    
        ### mirror statistics: influence function
        m = get_m(model1, model2, x_train1, x_train2)

        ### mirror statistics: weight multiplication
        # w1 = get_weight(model1)
        # w2 = get_weight(model2)
        # M = np.abs(w1+w2)-np.abs(w1-w2)
        # m = np.sum(M, 0)

        selected_index = analys(m, abs(m), q)
        if it == 0:
            DS_selected_index = selected_index
        num_select[it] = len(selected_index)
        if num_select[it] != 0:
            inclusion_rate[it, selected_index] = 1 / num_select[it]

    ### backtracking   
    inclusion_rate = np.mean(inclusion_rate, axis = 0)
    feature_rank = np.argsort(inclusion_rate)
    null_feature = []
    for feature_index in range(len(feature_rank)):
        if np.sum(inclusion_rate[feature_rank[:feature_index]]) > q:
            break
        else:
            null_feature.append(feature_rank[feature_index])
    MDS_selected_index = np.setdiff1d(feature_rank, null_feature)
    
    return [DS_selected_index, MDS_selected_index]



### loop over different p
DS_FDP, MDS_FDP = [], []
DS_POWER, MDS_POWER = [], []
for p in pset:
    ### L1-regularization magnitude
    ### power function: 1.0; exponential function: 0.01; sigmoid function: 0.01
    c = 1.0*math.sqrt(math.log(p)/n)
    ### neural network structure
    Hidden = [int(20*math.log(p)), int(10*math.log(p))]

    ### specify the covariance matrix
    ### case 1: power decay partial correlation
    Omega = np.mat(np.diag([1] * p), dtype = 'float32')
    for i in range(0, p):
        for j in range(0, p):
            Omega[i, j] = rho**abs(i-j)

    for i in range(0, p):
        Omega[i,i] = 1

    Sigma = 0.5 * (Omega.I + Omega.I.T)
    ### case 2: power decay correlation
    # Sigma = np.mat(np.diag([1] * p), dtype='float32')
    # for i in range(0, p):
    #     for j in range(0, p):
    #         Sigma[i, j] = rho**abs(i-j)

    # for i in range(0, p):
    #     Sigma[i,i] = 1

    z_train_original, nonzero, zero, x_train = grt(n = n, p = p, nz = nz, Sigma = Sigma)
    z_train = f(z_train_original) + np.random.randn(n)*ns
    DS_selected_index, MDS_selected_index = DS(x_train, z_train, q)
    ds_fdp, ds_power = fdp_power(DS_selected_index)
    mds_fdp, mds_power = fdp_power(MDS_selected_index)
    DS_FDP.append(ds_fdp)
    DS_POWER.append(ds_power)
    MDS_FDP.append(mds_fdp)
    MDS_POWER.append(mds_power)



