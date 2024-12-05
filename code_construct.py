import torch
from torch import nn
import torch.nn.functional as F
import scipy
import numpy as np


def Hardmard_code(num_class = 10, code_len=16, ser = None, is_sigmoid = False):
    hadamard = scipy.linalg.hadamard(code_len)
    if np.any(ser == None):
        # current_state = np.random.get_state()
        np.random.seed(4)
        ser = np.random.choice(code_len, num_class, replace = False)
        # np.random.set_state(current_state)
        # ser = np.arange(num_class)
    codebook = hadamard[ser]
    if is_sigmoid:
        codebook = (codebook+1)/2
    return torch.tensor(codebook, dtype = torch.float)


def Hardmard_code_list(num_class = 10, code_len=16, ser = None, is_sigmoid = False):
    hadamard = scipy.linalg.hadamard(code_len)
    # np.random.shuffle(hadamard)
    if np.any(ser == None):
        ser = np.arange(num_class)
    code_list = []
    for i in range(num_class // 2 + 1):
        code = np.concatenate([hadamard[0:i], -hadamard[0:num_class-i]])
        code_list.append(code)
    code_list = np.array(code_list)

    if is_sigmoid:
        code_list = (code_list+1)/2
    return torch.tensor(code_list, dtype = torch.float)



def opt_code(is_sigmoid):
    codebook = np.load('./test_code.npy')
    if is_sigmoid:
        codebook = (codebook+1)/2
    return torch.tensor(codebook, dtype = torch.float)    



def OneHot_code(num_class = 10, is_sigmoid = False):
    codebook = np.eye(num_class)
    if is_sigmoid:
        return torch.tensor(codebook, dtype = torch.float)
    else:
        return torch.tensor(codebook, dtype = torch.float)*2-1
    



def DNN_code_gen(num_class = 10, code_len=16, is_sigmoid = False, device = 'cpu', lam = 0.0):
    
    num_epochs = 30000
    enc = nn.Sequential(
        nn.Linear(num_class, 1024),
        torch.nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        torch.nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, code_len),
        nn.Tanh()
        ).train().to(device)
    optimizer = torch.optim.SGD(enc.parameters(), lr = 0.001, momentum=0.9)
    for ep in range(num_epochs):
        onehot_inputs = torch.eye(num_class).to(device)
        enc_outputs = enc(onehot_inputs)
        optimizer.zero_grad()
        cov = enc_outputs@enc_outputs.T
        loss_orth = torch.sum(torch.triu(cov, diagonal=1)**2)/code_len/num_class/(num_class-1) - 0.01*torch.sum((torch.diag(cov))**2)/num_class/code_len  
        loss = -torch.sum(F.pdist(enc_outputs, p=2))/num_class/(num_class-1)/code_len + lam*loss_orth
        loss.backward()
        optimizer.step()

    codes = torch.sign(enc_outputs).detach()
    if is_sigmoid:
        codes = (codes+1)/2

    return codes
    





